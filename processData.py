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

def jetIDCut(darkIDsToKeep,dataSet):
    jCatSeries = dataSet["jetCategory"]
    darkCond = (jCatSeries == -1) | (jCatSeries == -2) # these are the QCD and TTJets
    for darkID in darkIDsToKeep:
        darkCond |=  jCatSeries ==  darkID
    return darkCond

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

def getParticleNetInputs(dataSet,jetFeat,inputFileNames,numConst,pT,weight,jetCategory):
    varSet = dataSet.columns.tolist()
    data = dataSet.to_numpy()
    evtNumIndex = varSet.index("jCstEvtNum")
    fJetNumIndex = varSet.index("jCstJNum")
    etaIndex = varSet.index("deltaEta")
    phiIndex = varSet.index("deltaPhi")
    inFileIndex = varSet.index("inputFile")
    hvIndex = varSet.index("jetCategory")
    jIDIndex = varSet.index("jID")
    allJetCats = list(np.unique(jetCategory))
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
        pTs.append(float(pT.iloc[sInd:eInd].iloc[0]))
        weights.append(float(weight.iloc[sInd:eInd].iloc[0]))
        inputFileName = inputFileNames[int(sameJetConstData[0][inFileIndex])]
        getPara(inputFileName,"mMed",mMeds) 
        getPara(inputFileName,"mDark",mDarks)
        getPara(inputFileName,"rinv",rinvs)
        getPara(inputFileName,"alpha",alphas)
        jetCat = int(jetCategory.iloc[sInd:eInd].iloc[0])
        signalLabel = np.zeros(len(allJetCats))
        signalLabel[allJetCats.index(jetCat)] = 1
        signal.append(signalLabel)
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
        inputPoints.append(np.array(eachJetPoints,dtype=np.float32))
        inputFeatures.append(np.array(eachJetConstFeatures,dtype=np.float32))
        inputJetFeatures.append(np.array(eachJetJFeatures,dtype=np.float32))
        inputFileIndices.append(np.array(inFileColumn[sInd:eInd][0],dtype=np.float32))
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
    print("Splitting dataset into train, validation, test")
    for ind in uniqueIndices:
        inputFileNameOrder.append(inputFileNames[int(ind)])
        prng = np.random.RandomState(int(ind))
        inputIndices = np.where(inputFileIndices==ind)[0]
        print(inputFileNames[int(ind)])
        print("all jets",len(inputIndices))
        preTrain = len(train_indices)
        preTest = len(test_indices)
        preVal = len(validation_indices)
        inputIndices_len = len(inputIndices)
        prng.shuffle(inputIndices)
        train_len = int(inputIndices_len*sample_fractions[0])
        if (inputIndices_len - train_len) % 2 != 0:
            train_len += 1
        test_len =  int((inputIndices_len - train_len)/2)
        train_indices += list(inputIndices[:train_len])
        test_indices += list(inputIndices[train_len:train_len+test_len])
        validation_indices += list(inputIndices[train_len+test_len:])
        print("train jets",len(train_indices)-preTrain)
        print("test jets",len(test_indices)-preTest)
        print("validation jets",len(validation_indices)-preVal)
    return train_indices,test_indices,validation_indices,inputFileNameOrder

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
    inFileInds,inFileCounts = np.unique(inputFileIndices_subset,return_counts=True)
    for i in range(len(inFileInds)):
        inFileCount = inFileCounts[i]
        ind = inFileInds[i]
        dataInfo.append("{}:{} jets".format(np.array(inputFileNames)[int(ind)],inFileCount))
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
    fileIndex = 0
    inputFileNames = []
    variables = jetConstFeat + jetFeat + [weight_var]
    for key,fileList in samples.items():
        for fileName in fileList:
            print(fileName)
            f = up.open(inputFolder  + fileName + ".root")
            ftree = f[tree]
            branches = ftree.arrays(variables,library="pd")
            jetCat = [999]
            if key == "signal":
                jetCatBranch = ftree.arrays("jCsthvCategory",library="pd")
                jetCat = jetCatBranch["jCsthvCategory"]
            elif key == "QCD":
                jetCat = [-2]*len(branches)
            elif key == "TTJets":
                jetCat = [-1]*len(branches)
            branches["jetCategory"] = jetCat
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
            fileIndex += 1
    inputFileNames = np.array(inputFileNames)
    dataSet = pd.concat(dSets)
    print("dataSet jetCategory before removing SVJ_SM")
    print(dataSet["jetCategory"].value_counts())
    print("The number of constituents in each input training file before removing SVJ_SM:")
    unique, uniqueCounts = np.unique(dataSet["inputFile"],return_counts=True)
    for i in range(len(unique)):
        uInd = int(unique[i])
        print(inputFileNames[uInd],uniqueCounts[i])
    # combine different categories of SVJs
    categoryMerge = {
        # Q and QM
        3: 0,
        9: 1,
        # QM_G, G_Q, QM_Q, G
        11: 2,
        5: 3,
        7: 3,
        13: 3
        # The categories below are commented out so that we are not using them for training
        # SM, SMM
        #0: 2,
        #16: 2,
        # SMM_G, lD, SMM_lD
        #1: 3,
        #17: 3,
        #21: 3,
    }
    keepCond = jetIDCut(categoryMerge.keys(),dataSet)
    dataSet = dataSet[keepCond]
    dataSet.replace({"jetCategory":categoryMerge},inplace=True)
    # remove constituents with 0 pT or 0 energy
    dataSet = dataSet[dataSet["jCstPt"]>0]
    dataSet = dataSet[dataSet["jCstEnergy"]>0]
    print("dataSet jetCategory after removing SVJ_SM")
    print(dataSet["jetCategory"].value_counts())        
    print("The number of constituents in each input training file after removing SVJ_SM:")
    unique, uniqueCounts = np.unique(dataSet["inputFile"],return_counts=True)
    for i in range(len(unique)):
        uInd = int(unique[i])
    print(inputFileNames[uInd],uniqueCounts[i])
    raise Exception("Arretez")
    jetIdentifier(dataSet)
    dataSet.sort_values("jID",inplace=True)
    pT = dataSet[uniform_var]
    weight = dataSet[weight_var]
    jetCategory = dataSet["jetCategory"]
    print(dataSet.head())
    print("The number of constituents in each input training file:")
    print(dataSet["inputFile"].value_counts())
    dataSet["deltaEta"] = dataSet["jCstEtaAK8"] - dataSet["jCstEta"]
    dataSet["deltaPhi"] = deltaPhi(dataSet["jCstPhiAK8"],dataSet["jCstPhi"])
    dataSet["logPT"] = np.log(dataSet["jCstPt"])
    dataSet["logE"] = np.log(dataSet["jCstEnergy"])
    dataSet["logPTrpTJ"] = np.log(np.divide(dataSet["jCstPt"],dataSet["jCstPtAK8"]))
    dataSet["logErEJ"] = np.log(np.divide(dataSet["jCstEnergy"],dataSet["jCstEnergyAK8"]))
    dataSet["deltaR"] = np.sqrt(np.square(dataSet["deltaEta"]) + np.square(dataSet["deltaPhi"]))
    #dataSet["tanhdxy"] = np.tanh(dataSet["jCstdxy"])
    #dataSet["tanhdz"] = np.tanh(dataSet["jCstdz"])
    #dataSet = dataSet.drop(['jCstPt', 'jCstEta', 'jCstPhi', 'jCstEnergy', 'jCstPtAK8', 'jCstEtaAK8', 'jCstPhiAK8', 'jCstEnergyAK8', 'jCstdxy', 'jCstdz', weight_var],axis=1)
    dataSet = dataSet.drop(['jCstPt', 'jCstEta', 'jCstPhi', 'jCstEnergy', 'jCstPtAK8', 'jCstEtaAK8', 'jCstPhiAK8', 'jCstEnergyAK8', weight_var],axis=1)
    # one hot encoding pdgID
    dataSet["jCstPdgId"] = dataSet["jCstPdgId"].replace(np.unique(dataSet["jCstPdgId"]),["pdgId{}".format(int(pid)) for pid in np.unique(dataSet["jCstPdgId"])])
    dummies = pd.get_dummies(dataSet["jCstPdgId"],drop_first=True)
    dataSet = pd.concat([dataSet.drop("jCstPdgId",axis=1),dummies],axis=1)
    print("Order of variables:")
    for var in dataSet.columns:
        print(var)
    # getting input features into the right shape for particleNet
    inputPoints, inputFeatures, inputJetFeatures_unNormalized, inputFeaturesVarName, jetFeat, signal, inputFileIndices, pT, weight, mMed, mDark, rinv, alpha  = getParticleNetInputs(dataSet,jetFeat,inputFileNames,numConst,pT,weight,jetCategory)
    # splitting up the dataset into train, test, validation
    print("total jets", len(inputFileIndices))
    print("The number of jets before splitting:")
    unique, uniqueCounts = np.unique(inputFileIndices,return_counts=True)
    for i in range(len(unique)):
        uInd = int(unique[i])
        print(inputFileNames[uInd],uniqueCounts[i])    
    train_indices,test_indices,validation_indices,inputFileNameOrder = ttvIndex(inputFileIndices,sample_fractions,inputFileNames)
    # normalize jet variables and saving normalization information
    outputFolder = "processedDataNPZ"
    outputNPZFileName = "processedData_nc{}".format(numConst)
    jMean = np.mean(inputJetFeatures_unNormalized,axis=0)
    jStd = np.std(inputJetFeatures_unNormalized,axis=0)
    inputJetFeatures = (inputJetFeatures_unNormalized - jMean)/jStd
    inferenceDict = {
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

