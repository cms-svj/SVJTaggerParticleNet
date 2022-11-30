import numpy as np
import uproot as up
from magiconfig import ArgumentParser, MagiConfigOptions, ArgumentDefaultsRawHelpFormatter
from configs import configs as c
import torch.utils.data as udata
import torch
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

def pdgIDClass(idSeries):
    possibleIDs = [-211,-13,-11,1,2,11,13,22,130,211]
    #possibleIDs = [1,2,11,13,22,130,211]
    #idSeries = idSeries.abs()
    for i, iD in enumerate(possibleIDs):
        idSeries = idSeries.replace(iD,i)
    return idSeries

def phi_(x,y):
    phi = np.arctan2(y,x)
    return np.where(phi < 0, phi + 2*np.pi, phi)

def deltaPhi(phiVal1,phiVal2):
    phi1 = phi_( np.cos(phiVal1), np.sin(phiVal1) )
    phi2 = phi_( np.cos(phiVal2), np.sin(phiVal2) )
    dphi = phi1 - phi2
    dphi_edited = np.where(dphi < -np.pi, dphi + 2*np.pi, dphi)
    dphi_edited = np.where(dphi_edited > np.pi, dphi_edited - 2*np.pi, dphi_edited)
    return dphi_edited

def getPara(fileName,paraName,paraList):
    paravalue = 0
    if "SVJ" in fileName:
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
    paraList.append(paravalue)

def getParticleNetInputs(dataSet,jetFeat,inputFileNames,numConst,pT,weight):
    varSet = dataSet.columns.tolist()
    data = dataSet.to_numpy()
    evtNumIndex = varSet.index("jCstEvtNum")
    fJetNumIndex = varSet.index("jCstJNum")
    etaIndex = varSet.index("deltaEta")
    phiIndex = varSet.index("deltaPhi")
    inFileIndex = varSet.index("inputFile")
    hvIndex = varSet.index("jCsthvCategory")
    jIDIndex = varSet.index("jID")
    jetFeatIndices = []
    jetFeatLabels = []
    for jF in jetFeat:
        if jF not in ['jCstPtAK8', 'jCstEtaAK8', 'jCstPhiAK8', 'jCstEnergyAK8']:
            jetFeatIndices.append(varSet.index(jF))
            jetFeatLabels.append(jF)
    inFileColumn = data[:,inFileIndex]
    jIDColumn = data[:,jIDIndex]

    inputPoints = []
    inputFeatures = []
    inputFeaturesVarName = []
    inputJetFeatures = []
    inputFileIndices = []
    # grouping constituents that belong to the same jet together
    print("There are {} unique jets.".format(len(np.unique(jIDColumn))))
    count = 1
    signal = []
    pTs = []
    weights = []
    mMeds = []
    mDarks = []
    rinvs = []
    alphas = []
    jIDs, jIDCounts = np.unique(jIDColumn,return_counts=True) 
    jIDCounter = 0 
    # looping over jets
    for jIDCount in jIDCounts: 
        if count % 10000 == 0:
            print("Transformed {} jets".format(count))
        count += 1
        sInd = jIDCounter
        eInd = jIDCounter+jIDCount
        jIDCounter = eInd
        sameJetConstData = data[sInd:eInd] # getting values for constituents in the same jet
        pTs.append(float(pT[sInd:eInd].iloc[0]))
        weights.append(float(weight[sInd:eInd].iloc[0]))
        inputFileName = inputFileNames[int(sameJetConstData[0][inFileIndex])]
        getPara(inputFileName,"mMed",mMeds) 
        getPara(inputFileName,"mDark",mDarks)
        getPara(inputFileName,"rinv",rinvs)
        getPara(inputFileName,"alpha",alphas)
        if "QCD" in inputFileName:
            signal.append([1, 0, 0])
        elif "TTJets" in inputFileName:
            signal.append([0, 1, 0])
        else:
            signal.append([0, 0, 1])
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
            elif i not in [evtNumIndex,fJetNumIndex,inFileIndex,hvIndex,jIDIndex]:
                eachJetConstFeatures.append(paddedJetConstData[i])
                if count == 2:
                    inputFeaturesVarName.append(varSet[i])
        inputPoints.append(eachJetPoints)
        inputFeatures.append(eachJetConstFeatures)
        inputJetFeatures.append(eachJetJFeatures)
        inputFileIndices.append(inFileColumn[sInd:eInd][0])
    inputPoints = np.array(inputPoints)
    inputFeatures = np.array(inputFeatures)
    inputJetFeatures = np.array(inputJetFeatures)
    inputFeaturesVarName = np.array(inputFeaturesVarName)
    inputFileIndices = np.array(inputFileIndices)
    pTs = np.array(pTs)
    weights = np.array(weights)
    mMeds = np.array(mMeds)
    mDarks = np.array(mDarks)
    rinvs = np.array(rinvs)
    alphas = np.array(alphas)
    print("There are {} labels.".format(len(signal)))
    return inputPoints, inputFeatures, inputJetFeatures, inputFeaturesVarName, jetFeatLabels, signal, inputFileIndices, pTs, weights, mMeds, mDarks, rinvs, alphas

def normalize(df):
    return (df-df.mean())/df.std()

def jetIdentifier(dataSet):
    dataSet["jID"] = (dataSet["jCstEvtNum"].astype(int))*10**6 + (dataSet["inputFile"].astype(int))*1000 + dataSet["jCstJNum"].astype(int)

# this function randomly splits the dataset into train, test, val sets, but retain the same proportion of jets found in each sample
def ttvIndex(inputFileIndices,sample_fractions,inputFileNames):
    uniqueIndices = np.unique(inputFileIndices)
    train_indices = []
    test_indices = []
    validation_indices = []
    inputFileNameOrder = []
    test_scales = []
    for ind in uniqueIndices:
        inputFileNameOrder.append(inputFileNames[int(ind)])
        prng = np.random.RandomState(int(ind))
        inputIndices = np.where(inputFileIndices==ind)[0]
        print("len(inputIndices)",len(inputIndices))
        inputIndices_len = len(inputIndices)
        prng.shuffle(inputIndices)
        train_len = int(inputIndices_len*sample_fractions[0])
        if (inputIndices_len - train_len) % 2 != 0:
            train_len += 1
        test_len =  int((inputIndices_len - train_len)/2)
        print("train_len",train_len)
        print("test_len",test_len)
        train_indices += list(inputIndices[:train_len])
        test_indices += list(inputIndices[train_len:train_len+test_len])
        validation_indices += list(inputIndices[train_len+test_len:])
        test_scales.append(float(inputIndices_len)/test_len)
    return train_indices,test_indices,validation_indices,inputFileNameOrder,test_scales

def scale_weight(inputFileIndices_subset,inputFileIndices,weight):
    uniInput, uniCount = np.unique(inputFileIndices,return_counts=True)
    uniInput_sub, uniCount_sub = np.unique(inputFileIndices_subset,return_counts=True)
    scales = uniCount/uniCount_sub
    for i in range(len(uniInput_sub)):
        uIn_sub = uniInput_sub[i]
        scale = scales[i]
        weight[inputFileIndices_subset == uIn_sub] *= scale
    
def save_npz(dataset_indices, dataset_type, inputFileNames, inputPoints, inputFeatures, inputJetFeatures, inputFeaturesVarName, jetFeat, signal, inputFileIndices, pT, weight, mMed, mDark, rinv, alpha, outputFolder,outputNPZFileName):
    inputPoints = np.take(inputPoints,dataset_indices,axis=0)
    inputFeatures = np.take(inputFeatures,dataset_indices,axis=0)
    inputJetFeatures = np.take(inputJetFeatures,dataset_indices,axis=0)
    inputFileIndices_subset = np.take(inputFileIndices,dataset_indices,axis=0)
    signal = np.take(signal,dataset_indices,axis=0)
    pT = np.take(pT,dataset_indices,axis=0)
    weight = np.take(weight,dataset_indices,axis=0)
    scale_weight(inputFileIndices_subset,inputFileIndices,weight)
    mMed = np.take(mMed,dataset_indices,axis=0)
    mDark = np.take(mDark,dataset_indices,axis=0)
    rinv = np.take(rinv,dataset_indices,axis=0)
    alpha = np.take(alpha,dataset_indices,axis=0)
    dataInfo = []
    dataInfo.append("The number of jets from each file:")
    inFileInds,inFileCounts = np.unique(inputFileIndices,return_counts=True)
    for i in inFileInds:
        inFileCount = inFileCounts[int(i)]
        dataInfo.append("{}:{} jets".format(np.array(inputFileNames)[int(i)],inFileCount))
    with open('{}/{}_{}_dataInfo.txt'.format(outputFolder,outputNPZFileName,dataset_type), 'w') as f:
        for line in dataInfo:
            f.write("{}\n".format(line))
    dictOut = {
        "inputPoints":inputPoints,
        "inputFeatures":inputFeatures,
        "inputJetFeatures":inputJetFeatures,
        "inputFileIndices":inputFileIndices_subset,
        "signal":signal,
        "pT":pT,
        "weight":weight,
        "mMed":mMed,
        "mDark":mDark,
        "rinv":rinv,
        "alpha":alpha,
        "inputFileNames":inputFileNames,
        "inputFeaturesVarName":inputFeaturesVarName
    }
    print("pT",len(dictOut['pT']))
    print("weight",len(dictOut['weight']))
    print("signal",len(dictOut['signal']))
    print("inputFileIndices",len(dictOut['inputFileIndices']))    
    print("inputFileNames",inputFileNames)
    print("inputPoints",dictOut['inputPoints'].shape)
    print("inputFeatures",dictOut['inputFeatures'].shape)
    print("inputJetFeatures",dictOut['inputJetFeatures'].shape)  
    print("mMed", np.unique(dictOut["mMed"]))      
    print("mDark", np.unique(dictOut["mDark"]))      
    print("rinv", np.unique(dictOut["rinv"]))      
    print("alpha", np.unique(dictOut["alpha"]))      
    np.savez_compressed("{}/{}_{}.npz".format(outputFolder,outputNPZFileName,dataset_type),**dictOut)

def process_all_vars(inputFolder, samples, jetConstFeat, jetFeat, pTBins, uniform_var, weight_var, numConst, num_classes, sample_fractions, tree="tree"):
    dSets = []
    pTLab = np.array([])
    fileIndex = 0
    inputFileNames = []
    variables = jetConstFeat + jetFeat + [weight_var]
    for key,fileList in samples.items():
        nsigfiles = len(samples["signal"])
        nbkgfiles = len(samples["background"])
        for fileName in fileList:
            print(fileName)
            f = up.open(inputFolder  + fileName + ".root")
            ftree = f[tree]
            branches = ftree.arrays(variables,library="pd")
            if key == "signal":
                jetCatBranch = ftree.arrays("jCsthvCategory",library="pd")
                darkCon = ((jetCatBranch["jCsthvCategory"] == 3) | (jetCatBranch["jCsthvCategory"] == 5) | (jetCatBranch["jCsthvCategory"] == 9))
                branches = branches[darkCon]
            inputFileNames.append(fileName)
            branches["inputFile"] = [fileIndex]*len(branches) # record name of the input file, important for distinguishing which jet the constituents belong to
            #branches.replace([np.inf, -np.inf], np.nan, inplace=True)
            #branches = branches.dropna()
            numEvent = len(branches)
            #if key == "signal":
            #    numEvent = 10000#500000#105238,500000
            #else:
            #    numEvent = 10000#750000#150000,750000
            #numEvent = 1000            
            #branches = branches.head(numEvent)
            print("Total Number of constituents for {}".format(fileName))
            print(len(branches))
            # print(len(branches))
            # branches = branches.head(10000)
            dSets.append(branches)
            # get the pT label based on what pT bin the jet pT falls into
            branch = ftree.arrays(uniform_var,library="pd")
            branch = branch.head(len(branches)).to_numpy().flatten()
            pTLabel = np.digitize(branch,pTBins) - 1.0
            pTLab = np.append(pTLab,pTLabel)
            fileIndex += 1
    inputFileNames = np.array(inputFileNames)
    dataSet = pd.concat(dSets)
    jetIdentifier(dataSet)
    dataSet.sort_values("jID",inplace=True)
    pT = dataSet[uniform_var]
    weight = dataSet[weight_var]
    #dataSet["jCstPdgId"] = pdgIDClass(dataSet["jCstPdgId"])
    print(dataSet.head())
    print("The number of constituents in each input training file:")
    print(dataSet["inputFile"].value_counts())
    dataSet["deltaEta"] = dataSet["jCstEtaAK8"] - dataSet["jCstEta"]
    dataSet["deltaPhi"] = deltaPhi(dataSet["jCstPhiAK8"],dataSet["jCstPhi"])
    cond = (np.array(dataSet["jCstPhiAK8"] - dataSet["jCstPhi"]) < -np.pi) | (np.array(dataSet["jCstPhiAK8"] - dataSet["jCstPhi"]) > np.pi)
    dataSet["logPT"] = np.log(dataSet["jCstPt"])
    dataSet["logE"] = np.log(dataSet["jCstEnergy"])
    dataSet["logPTrpTJ"] = np.log(np.divide(dataSet["jCstPt"],dataSet["jCstPtAK8"]))
    dataSet["logErEJ"] = np.log(np.divide(dataSet["jCstEnergy"],dataSet["jCstEnergyAK8"]))
    dataSet["deltaR"] = np.sqrt(np.square(dataSet["deltaEta"]) + np.square(dataSet["deltaPhi"]))
    dataSet["tanhdxy"] = np.tanh(dataSet["jCstdxy"])
    dataSet["tanhdz"] = np.tanh(dataSet["jCstdz"])
    dataSet = dataSet.drop(['jCstPt', 'jCstEta', 'jCstPhi', 'jCstEnergy', 'jCstPtAK8', 'jCstEtaAK8', 'jCstPhiAK8', 'jCstEnergyAK8', 'jCstdxy', 'jCstdz', weight_var],axis=1)
    columns_to_normalize = ['jCstPdgId']
    dataSetToNormalize = dataSet[columns_to_normalize]
    jConstmean = dataSetToNormalize.mean()
    jConststd = dataSetToNormalize.std()
    dataSet[columns_to_normalize] = normalize(dataSetToNormalize)
    print("Order of variables:")
    for var in dataSet.columns:
        print(var)
    # getting input features into the right shape for particleNet
    inputPoints, inputFeatures, inputJetFeatures_unNormalized, inputFeaturesVarName, jetFeat, signal, inputFileIndices, pT, weight, mMed, mDark, rinv, alpha  = getParticleNetInputs(dataSet,jetFeat,inputFileNames,numConst,pT,weight)
    # splitting up the dataset into train, test, validation
    print("total jets", len(inputFileIndices))
    train_indices,test_indices,validation_indices,inputFileNameOrder,test_scales = ttvIndex(inputFileIndices,sample_fractions,inputFileNames)
    # normalize jet variables and saving normalization information
    outputFolder = "processedDataNPZ"
    outputNPZFileName = "processedData_nc{}".format(numConst)
    jMean = np.mean(inputJetFeatures_unNormalized,axis=0)
    jStd = np.std(inputJetFeatures_unNormalized,axis=0)
    inputJetFeatures = (inputJetFeatures_unNormalized - jMean)/jStd
    inferenceDict = {
        "jConstVariables":columns_to_normalize,
        "jConstmean":jConstmean,
        "jConststd":jConststd,
        "jVariables":jetFeat,
        "jMean":jMean,
        "jStd":jStd,
    }
    np.savez_compressed("{}/{}_Inference.npz".format(outputFolder,outputNPZFileName),**inferenceDict)
    # save input features as train, test, and validation sets
    save_npz(train_indices, "train", inputFileNames, inputPoints, inputFeatures, inputJetFeatures, inputFeaturesVarName, jetFeat, signal, inputFileIndices, pT, weight, mMed, mDark, rinv, alpha, outputFolder,outputNPZFileName)
    save_npz(test_indices, "test", inputFileNames, inputPoints, inputFeatures, inputJetFeatures, inputFeaturesVarName, jetFeat, signal, inputFileIndices, pT, weight, mMed, mDark, rinv, alpha, outputFolder,outputNPZFileName)
    save_npz(validation_indices, "validation", inputFileNames, inputPoints, inputFeatures, inputJetFeatures, inputFeaturesVarName, jetFeat, signal, inputFileIndices, pT, weight, mMed, mDark, rinv, alpha, outputFolder,outputNPZFileName)
   
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
    sample_fractions = args.dataset.sample_fractions
    jetConstFeat = args.features.jetConst
    jetFeat = args.features.jetVariables
    pTBins = args.hyper.pTBins
    uniform_var = args.features.uniform
    weight_var = args.features.weight
    numConst = args.hyper.numConst
    num_classes = args.hyper.num_classes
    if not os.path.isdir("processedDataNPZ"):
    	os.makedirs("processedDataNPZ")
    process_all_vars(dSet.path, inputFiles, jetConstFeat, jetFeat, pTBins, uniform_var, weight_var, numConst, num_classes, sample_fractions)

