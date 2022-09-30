import numpy as np
import uproot as up
from magiconfig import ArgumentParser, MagiConfigOptions, ArgumentDefaultsRawHelpFormatter
from configs import configs as c
import torch.utils.data as udata
import torch
import pandas as pd
from tqdm import tqdm
import os

def pdgIDClass(idSeries):
    possibleIDs = [-211,-13,-11,1,2,11,13,22,130,211]
    #possibleIDs = [1,2,11,13,22,130,211]
    #idSeries = idSeries.abs()
    for i, iD in enumerate(possibleIDs):
        idSeries = idSeries.replace(iD,i)
    return idSeries

def getParticleNetInputs(dataSet,jetFeat,signalFileIndex,numConst,signals,pT,weight,mcType):
    varSet = dataSet.columns.tolist()
    data = dataSet.to_numpy()
    evtNumIndex = varSet.index("jCstEvtNum")
    fJetNumIndex = varSet.index("jCstJNum")
    etaIndex = varSet.index("jCstEta")
    phiIndex = varSet.index("jCstPhi")
    inFileIndex = varSet.index("inputFile")
    hvIndex = varSet.index("jCsthvCategory")
    jIDIndex = varSet.index("jID")
    jetFeatIndices = [varSet.index(jF) for jF in jetFeat]
    inFileColumn = data[:,inFileIndex]
    jIDColumn = data[:,jIDIndex]

    inputPoints = []
    inputFeatures = []
    inputJetFeatures = []
    inputFileIndices = []
    # grouping constituents that belong to the same jet together
    print("There are {} unique jets.".format(len(np.unique(jIDColumn))))
    count = 1
    signal = []
    pTs = []
    weights = []
    mcTypes = []
    jIDs, jIDCounts = np.unique(jIDColumn,return_counts=True)
    jIDCounter = 0
    # looping over jets
    for jIDCount in jIDCounts:
        if count % 5000 == 0:
            print("Transformed {} jets".format(count))
        count += 1
        sInd = jIDCounter
        eInd = jIDCounter+jIDCount
        jIDCounter = eInd
        sameJetConstData = data[sInd:eInd] # getting values for constituents in the same jet
        pTs.append(float(pT[sInd:eInd].iloc[0]))
        weights.append(float(weight[sInd:eInd].iloc[0]))
        mcTypes.append(int(mcType[sInd:eInd][0]))
        if sameJetConstData[0][inFileIndex] in signalFileIndex:
            #signal.append([0, 1])
            signal.append([0, 0, 1])
        else:
            signal.append(signals[sInd:eInd][0])
        sameJetConstDataTr = np.transpose(sameJetConstData)
        if numConst > sameJetConstDataTr.shape[1]:
            paddedJetConstData = np.pad(sameJetConstDataTr,((0,0),(0,numConst-sameJetConstDataTr.shape[1])), 'constant', constant_values=0)
        else:
            paddedJetConstData = sameJetConstDataTr[:,:numConst]
        eachJetPoints = np.array([paddedJetConstData[etaIndex],paddedJetConstData[phiIndex]])
        eachJetConstFeatures = []
        eachJetJFeatures = []
        # looping over jet/constituent features
        for i in range(paddedJetConstData.shape[0]):
            # make sure information that would easily give away the identity of the jet is not included as input features
            if i in jetFeatIndices:
                eachJetJFeatures.append(paddedJetConstData[i][0])
            elif i not in [etaIndex,phiIndex,evtNumIndex,fJetNumIndex,inFileIndex,hvIndex,jIDIndex]:
                eachJetConstFeatures.append(paddedJetConstData[i])
        inputPoints.append(eachJetPoints)
        inputFeatures.append(eachJetConstFeatures)
        inputJetFeatures.append(eachJetJFeatures)
        inputFileIndices.append(inFileColumn[sInd:eInd][0])
    inputPoints = np.array(inputPoints)
    inputFeatures = np.array(inputFeatures)
    inputJetFeatures = np.array(inputJetFeatures)
    print("There are {} labels.".format(len(signal)))
    print(inputPoints.shape)
    print(inputFeatures.shape)
    return inputPoints, inputFeatures, inputJetFeatures, signal, inputFileIndices, pTs, weights, mcTypes

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
        elif paraName == "rinv":
            paravalue = float(paravalue.replace("p","."))
        else:
            paravalue = float(paravalue)
    paraList += [paravalue]*len(branches)

def normalize(df):
    return (df-df.mean())/df.std()

def jetIdentifier(dataSet):
    dataSet["jID"] = (dataSet["jCstEvtNum"].astype(int))*10**6 + (dataSet["inputFile"].astype(int))*1000 + dataSet["jCstJNum"].astype(int)

def process_all_vars(inputFolder, samples, jetConstFeat, jetFeat, pTBins, uniform, mT, weight, numConst, num_classes, tree="tree"):
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
    inputFileNames = []
    signalFileIndex = []
    variables = jetConstFeat + jetFeat
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
                branches = branches[darkCon]
            inputFileNames.append(fileName)
            branches["inputFile"] = [fileIndex]*len(branches) # record name of the input file, important for distinguishing which jet the constituents belong to
            fileIndex += 1
            branches.replace([np.inf, -np.inf], np.nan, inplace=True)
            branches = branches.dropna()
            numEvent = len(branches)
            # if we do not limit the number of constituents we read in, the code is gonna take very long to run
            #if key == "signal":
            #    numEvent = 10000#500000#105238,500000
            #else:
            #    numEvent = 10000#750000#150000,750000
            #branches = branches.head(numEvent)
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
                #signal += list([1, 0] for _ in range(len(branches)))
                signal += list([1, 0, 0] for _ in range(len(branches)))
                if fileName == "tree_SVJ_mZprime-3000_mDark-20_rinv-0.3_alpha-peak_MC2017":
                    mcType += [1] * len(branches)
                else:
                    mcType += [0] * len(branches)
            else:
                #signal += list([1, 0] for _ in range(len(branches)))
                if "QCD" in fileName:
                    #signal += list([1, 0] for _ in range(len(branches)))
                    signal += list([1, 0, 0] for _ in range(len(branches)))
                    mcType += [2] * len(branches)
                else:
                    #signal += list([1, 0] for _ in range(len(branches)))
                    signal += list([0, 1, 0] for _ in range(len(branches)))
                    mcType += [3] * len(branches)
            print("Number of Constituents",len(branches))
    mcType = np.array(mcType)
    mMed = np.array(mMeds)
    mDark = np.array(mDarks)
    rinv = np.array(rinvs)
    alpha = np.array(alphas)
    dataSet = pd.concat(dSets)
    jetIdentifier(dataSet)
    dataSet.sort_values("jID",inplace=True)
    pT = pd.concat(pTs)
    mT = pd.concat(mTs)
    weight = pd.concat(weights)
    print("weight")
    print(np.unique(weight))
    #dataSet["jCstPdgId"] = pdgIDClass(dataSet["jCstPdgId"])
    print(dataSet.head())
    print("The number of constituents in each input training file:")
    print(dataSet["inputFile"].value_counts())
    
    dataSet["jCstEta_Norm"] = dataSet["jCstEta"]
    dataSet["jCstPhi_Norm"] = dataSet["jCstPhi"]
    allVariables = dataSet.columns
    columns_to_normalize = [var for var in allVariables if var not in ["jCstEta","jCstPhi","inputFile","jCstEvtNum","jCstJNum"] + jetFeat]
    dataSetToNormalize = dataSet[columns_to_normalize]
    jConstmean = dataSetToNormalize.mean()
    jConststd = dataSetToNormalize.std()
    dataSet[columns_to_normalize] = normalize(dataSetToNormalize)
    print("Order of variables:")
    for var in dataSet.columns:
        print(var)
    inputPoints, inputFeatures, inputJetFeatures_unNormalized, signal, inputFileIndices, pT, weight, mcType = getParticleNetInputs(dataSet,jetFeat,signalFileIndex,numConst,signal,pT,weight,mcType)
    sigLabel = np.array(signal)[:,num_classes-1]
    outputFolder = "processedDataNPZ"
    outputNPZFileName = "processedData_nc{}".format(numConst)
    dataInfo = []
    dataInfo.append("The total number of jets: {}".format(len(sigLabel)))
    dataInfo.append("Total number of signal jets: {}".format(len(sigLabel[sigLabel==1])))
    dataInfo.append("Total number of background jets: {}".format(len(sigLabel[sigLabel==0])))
    dataInfo.append("The number of jets from each file:")
    inFileInds,inFileCounts = np.unique(inputFileIndices,return_counts=True)
    for i in inFileInds:
        inFileCount = inFileCounts[int(i)]
        dataInfo.append("{}:{} jets".format(np.array(inputFileNames)[int(i)],inFileCount))
    with open('{}/{}_dataInfo.txt'.format(outputFolder,outputNPZFileName), 'w') as f:
        for line in dataInfo:
            f.write("{}\n".format(line))
    jMean = np.mean(inputJetFeatures_unNormalized,axis=0)
    jStd = np.std(inputJetFeatures_unNormalized,axis=0)
    inputJetFeatures = (inputJetFeatures_unNormalized - jMean)/jStd
    dictOut = {
        "inputPoints":inputPoints,
        "inputFeatures":inputFeatures,
        "inputJetFeatures":inputJetFeatures,
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
        "inputFileIndices":inputFileIndices,
        "signalFileIndex":signalFileIndex
    }
    inferenceDict = {
        "jConstVariables":columns_to_normalize,
        "jConstmean":jConstmean,
        "jConststd":jConststd,
        "jVariables":jetFeat,
        "jMean":jMean,
        "jStd":jStd,
    }
    print("pT",len(pT))
    print("weight",len(weight))
    print("jet constituent variables that are normalized",columns_to_normalize)
    print("jet variables that are normalized",jetFeat)
    print("inputPoints",inputPoints.shape)
    print("inputFeatures",inputFeatures.shape)
    print("inputJetFeatures",inputJetFeatures.shape)        
    np.savez_compressed("{}/{}.npz".format(outputFolder,outputNPZFileName),**dictOut)
    np.savez_compressed("{}/{}_Inference.npz".format(outputFolder,outputNPZFileName),**inferenceDict)
    
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
    jetConstFeat = args.features.jetConst
    jetFeat = args.features.jetVariables
    pTBins = args.hyper.pTBins
    uniform = args.features.uniform
    mTs = args.features.mT
    weights = args.features.weight
    numConst = args.hyper.numConst
    num_classes = args.hyper.num_classes
    if not os.path.isdir("processedDataNPZ"):
    	os.makedirs("processedDataNPZ")
    process_all_vars(dSet.path, inputFiles, jetConstFeat, jetFeat, pTBins, uniform, mTs, weights, numConst, num_classes)

