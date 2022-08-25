import numpy as np
import uproot as up
from magiconfig import ArgumentParser, MagiConfigOptions, ArgumentDefaultsRawHelpFormatter
from configs import configs as c
import torch.utils.data as udata
import torch
import pandas as pd
from tqdm import tqdm
import os

def getParticleNetInputs(dataSet,signalFileIndex,numConst,pT):
    varSet = dataSet.columns.tolist()
    data = dataSet.to_numpy()
    evtNumIndex = varSet.index("jCstEvtNum")
    fJetNumIndex = varSet.index("jCstJNum")
    etaIndex = varSet.index("jCstEta")
    phiIndex = varSet.index("jCstPhi")
    inFileIndex = varSet.index("inputFile")
    hvIndex = varSet.index("jCsthvCategory")
    jIDIndex = varSet.index("jID")
    evtNumColumn = data[:,evtNumIndex]
    fJetNumColumn = data[:,fJetNumIndex]
    inFileColumn = data[:,inFileIndex]
    inFColumn = data[:,inFileIndex]
    jIDColumn = data[:,jIDIndex]

    inputPoints = []
    inputFeatures = []
    inputFileIndices = []
    # grouping constituents that belong to the same jet together
    print("There are {} unique jets.".format(len(np.unique(jIDColumn))))
    count = 1
    signal = []
    pTs = []
    
    for jID in np.unique(jIDColumn):
        if count % 200 == 0:
            print("Transformed {} jets".format(count))
        count += 1
        sameJetConstData = data[jIDColumn == jID] # getting values for constituents in the same jet
        pTs.append(float(pT[jIDColumn == jID].iloc[0]))
        if sameJetConstData[0][inFileIndex] in signalFileIndex:
            signal.append([0, 1])
        else:
            signal.append([1, 0])
        sameJetConstDataTr = np.transpose(sameJetConstData)
        if numConst > sameJetConstDataTr.shape[1]:
            paddedJetConstData = np.pad(sameJetConstDataTr,((0,0),(0,numConst-sameJetConstDataTr.shape[1])), 'constant', constant_values=0)
        else:
            paddedJetConstData = sameJetConstDataTr[:,:numConst]
        eachJetPoints = np.array([paddedJetConstData[etaIndex],paddedJetConstData[phiIndex]])
        eachJetFeatures = []
        for i in range(paddedJetConstData.shape[0]):
            # make sure information that would easily give away the identity of the jet is not included as input features
            if i not in [etaIndex,phiIndex,evtNumIndex,fJetNumIndex,inFileIndex,hvIndex,jIDIndex]:
                eachJetFeatures.append(paddedJetConstData[i])
        inputPoints.append(eachJetPoints)
        inputFeatures.append(eachJetFeatures)
        inputFileIndices.append(inFileColumn[jIDColumn == jID][0])
    inputPoints = np.array(inputPoints)
    inputFeatures = np.array(inputFeatures)
    print("There are {} labels.".format(len(signal)))
    print(inputPoints.shape)
    print(inputFeatures.shape)
    return inputPoints, inputFeatures, signal, inputFileIndices, pTs

def getBranch(ftree,variable,branches,branchList):
    branch = ftree.arrays(variable,library="pd")
    branch = branch.head(len(branches))
    branchList.append(branch)

def getPara(fileName,paraName,paraList,branches,key):
    paravalue = 0
    if key == "signal":
        ind = fileName.find(paraName)
        fnCut = fileName[ind:]
        indUnd = fnCut.find("_")
        paravalue = fnCut[len(paraName)+1:indUnd]
        if paraName == "alpha":
            if paravalue == "low":
                paravalue = 1
            elif paravalue == "peak":
                paravalue = 2
            elif paravalue == "high":
                paravalue = 3
        else:
            paravalue = float(paravalue)
    paraList += [paravalue]*len(branches)

def normalize(df):
    return (df-df.mean())/df.std()

def jetIdentifier(dataSet):
    dataSet["jID"] = (dataSet["jCstEvtNum"].astype(int))*10**6 + (dataSet["inputFile"].astype(int))*1000 + dataSet["jCstJNum"].astype(int)

def process_all_vars(inputFolder, samples, variables, pTBins, uniform, mT, weight, numConst, tree="tree"):
    dSets = []
    signal = []
    mcType =[] # 0 = signals other than baseline, 1 = baseline signal, 2 = QCD, 3 = TTJets
    pTLab = np.array([])
    pTs = []
    mTs = []
    mMeds = []
    mDarks = []
    rinvs = []
    alphas = []
    weights = []
    fileIndex = 0
    signalFileIndex = []
    for key,fileList in samples.items():
        nsigfiles = len(samples["signal"])
        nbkgfiles = len(samples["background"])
        for fileName in fileList:
            print(fileName)
            f = up.open(inputFolder  + fileName + ".root")
            ftree = f[tree]
            branches = ftree.arrays(variables,library="pd")
            if key == "signal":
                signalFileIndex.append(fileIndex)
                jetCatBranch = ftree.arrays("jCsthvCategory",library="pd")
                darkCon = ((jetCatBranch["jCsthvCategory"] == 3) | (jetCatBranch["jCsthvCategory"] == 5) | (jetCatBranch["jCsthvCategory"] == 9))
                # print(jetCatBranch['jCsthvCategory'].value_counts())
                branches = branches[darkCon]
            branches["inputFile"] = [fileIndex]*len(branches) # record name of the input file, important for distinguishing which jet the constituents belong to
            fileIndex += 1
            branches.replace([np.inf, -np.inf], np.nan, inplace=True)
            branches = branches.dropna()
            numEvent = len(branches)
            # if we do not limit the number of constituents we read in, the code is gonna take very long to run
            if key == "signal":
                numEvent = 105238#105238
            else:
                numEvent = 150000#150000
            branches = branches.head(numEvent)
            print("Total Number of constituents for {}".format(fileName))
            print(len(branches))
            # print(len(branches))
            # branches = branches.head(10000)
            dSets.append(branches)
            getBranch(ftree,uniform,branches,pTs)
            getBranch(ftree,mT,branches,mTs)
            getBranch(ftree,weight,branches,weights)
            # getPara(fileName,"mZprime",mMeds,branches,key)
            getPara(fileName,"mMed",mMeds,branches,key) # use this for t-channel
            getPara(fileName,"mDark",mDarks,branches,key)
            getPara(fileName,"rinv",rinvs,branches,key)
            getPara(fileName,"alpha",alphas,branches,key)
            # get the pT label based on what pT bin the jet pT falls into
            branch = ftree.arrays(uniform,library="pd")
            branch = branch.head(len(branches)).to_numpy().flatten()
            pTLabel = np.digitize(branch,pTBins) - 1.0
            pTLab = np.append(pTLab,pTLabel)
            if key == "signal":
                signal += list([0, 1] for _ in range(len(branches)))
                if fileName == "tree_SVJ_mZprime-3000_mDark-20_rinv-0.3_alpha-peak_MC2017":
                    mcType += [1] * len(branches)
                else:
                    mcType += [0] * len(branches)
            else:
                signal += list([1, 0] for _ in range(len(branches)))
                if "QCD" in fileName:
                    mcType += [2] * len(branches)
                else:
                    mcType += [3] * len(branches)
            print("Number of Constituents",len(branches))
    mcType = np.array(mcType)
    mMed = np.array(mMeds)
    mDark = np.array(mDarks)
    rinv = np.array(rinvs)
    alpha = np.array(alphas)
    dataSet = pd.concat(dSets)
    jetIdentifier(dataSet)
    pT = pd.concat(pTs)
    mT = pd.concat(mTs)
    weight = pd.concat(weights)
    print("dataSet.head()")
    print("dataSet length",len(dataSet))
    print("pT length",len(pT))
    print(dataSet.head())
    print("The number of constituents in each input training file:")
    print(dataSet["inputFile"].value_counts())
    dfmean = dataSet.mean()
    dfstd = dataSet.std()
    dataSet["jCstEta_Norm"] = dataSet["jCstEta"]
    dataSet["jCstPhi_Norm"] = dataSet["jCstPhi"]
    columns_to_normalize = [var for var in variables if var not in ["jCstEta","jCstPhi","inputFile","jCstEvtNum","jCstJNum"]]
    dataSet[columns_to_normalize] = normalize(dataSet[columns_to_normalize])
    inputPoints, inputFeatures, signal, inputFileIndices, pT = getParticleNetInputs(dataSet,signalFileIndex,numConst,pT)
    sigLabel = np.array(signal)[:,1]
    print("Total number of pT",len(pT))
    print("The total number of jets: {}".format(len(sigLabel)))
    print("Total number of signal jets: {}".format(len(sigLabel[sigLabel==1])))
    print("Total number of background jets: {}".format(len(sigLabel[sigLabel==0])))
    
    dictOut = {
        "inputPoints":inputPoints,
        "inputFeatures":inputFeatures,
        "signal":signal,
        "mcType":mcType,
        "pTLab":pTLab,
        "pT":pT,
        "mT":mT,
        "weight":weight,
        "mMed":mMed,
        "mDark":mDark,
        "rinv":rinv,
        "alpha":alpha,
        "dfmean":dfmean,
        "dfstd":dfstd,
        "inputFileIndices":inputFileIndices,
        "signalFileIndex":signalFileIndex
    }

    np.savez_compressed("processedDataNPZ/processedData_nc{}.npz".format(numConst),**dictOut)

if __name__=="__main__":
    # parse arguments
    parser = ArgumentParser(config_options=MagiConfigOptions(strict = True, default="configs/C1.py"),formatter_class=ArgumentDefaultsRawHelpFormatter)
    parser.add_config_only(*c.config_schema)
    parser.add_config_only(**c.config_defaults)
    args = parser.parse_args()
    dSet = args.dataset
    sigFiles = dSet.signal
    inputFiles = dSet.background
    inputFiles.update(sigFiles)
    varSet = args.features.train
    pTBins = args.hyper.pTBins
    uniform = args.features.uniform
    mTs = args.features.mT
    weights = args.features.weight
    numConst = args.hyper.numConst
    if not os.path.isdir("processedDataNPZ"):
    	os.makedirs("processedDataNPZ")
    process_all_vars(dSet.path, inputFiles, varSet, pTBins, uniform, mTs, weights, numConst)

