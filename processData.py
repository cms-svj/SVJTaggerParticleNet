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
from sklearn.model_selection import train_test_split

def getPara(fileName,paraName):
    paravalue = 0
    if "SVJ" in fileName:
        ind = fileName.find(paraName)
        fnCut = fileName[ind:]
        indUnd = fnCut.find("_")
        paravalue = fnCut[len(paraName)+1:indUnd]
        if paraName == "rinv":
            paravalue = float(paravalue.replace("p","."))
        else:
            paravalue = float(paravalue)
    return paravalue

def scale_weight(inputFileIndices_subset,inputFileIndices,weight):
    uniInput, uniCount = np.unique(inputFileIndices,return_counts=True)
    uniInput_sub, uniCount_sub = np.unique(inputFileIndices_subset,return_counts=True)
    scales = uniCount/uniCount_sub
    for i in range(len(uniInput_sub)):
        uIn_sub = uniInput_sub[i]
        scale = scales[i]
        weight[inputFileIndices_subset == uIn_sub] *= scale
    
def save_npz(dataset_indices, dataset_type, inputFileNames, inputPoints, inputFeatures, masks, signal, inputFileIndices, pT, weight, mMed, mDark, rinv, jetConstFeat, outputFolder,outputNPZFileName):
    inputPoints = np.take(inputPoints,dataset_indices,axis=0)
    inputFeatures = np.take(inputFeatures,dataset_indices,axis=0)
    masks = np.take(masks,dataset_indices,axis=0)
    inputFileIndices_subset = np.take(inputFileIndices,dataset_indices,axis=0)
    signal = np.take(signal,dataset_indices,axis=0)
    pT = np.take(pT,dataset_indices,axis=0)
    weight = np.take(weight,dataset_indices,axis=0)
    scale_weight(inputFileIndices_subset,inputFileIndices,weight)
    mMed = np.take(mMed,dataset_indices,axis=0)
    mDark = np.take(mDark,dataset_indices,axis=0)
    rinv = np.take(rinv,dataset_indices,axis=0)
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
        "masks":masks,
        "inputFileIndices":inputFileIndices_subset,
        "signal":signal,
        "pT":pT,
        "weight":weight,
        "mMed":mMed,
        "mDark":mDark,
        "rinv":rinv,
        "inputFileNames":inputFileNames,
        "inputFeaturesVarName":jetConstFeat,
    }
    print("pT",len(dictOut['pT']))
    print("weight",len(dictOut['weight']))
    print("signal",len(dictOut['signal']))
    print("inputFileIndices",len(dictOut['inputFileIndices']))    
    print("inputFileNames",inputFileNames)
    print("inputPoints",dictOut['inputPoints'].shape)
    print("inputFeatures",dictOut['inputFeatures'].shape)
    print("mMed", np.unique(dictOut["mMed"]))      
    print("mDark", np.unique(dictOut["mDark"]))      
    print("rinv", np.unique(dictOut["rinv"]))      
    np.savez_compressed("{}/{}_{}.npz".format(outputFolder,outputNPZFileName,dataset_type),**dictOut)

def process_all_vars(inputFolder, samples, jetConstFeat, num_const, num_classes, sample_fractions, tree="tree"):
    dSets = []
    fileIndex = 0
    inputFileNames = []
    num_features = len(jetConstFeat)
    allPoints = np.empty((0,2,num_const))
    allFeatures = np.empty((0,num_features,num_const))
    allMask = np.empty((0,1,num_const))
    allPt = np.empty((0))
    allWeight = np.empty((0))
    allSignal = np.empty((0,num_classes))
    allFileLabel = []
    allMmed = []
    allMdark = []
    allRinv = []
    inputFileIndices = []
    inputFileNames = []
    inputFileIndex = 0
    totalNumOfJets = 0
    for key,fileList in samples.items():
        for fileName in fileList:
            print(fileName)
            inputFileNames.append(fileName)
            tr = up.open(inputFolder  + fileName + ".root:{}".format(tree))
            points = tr["points"].array(library="np")
            features = tr["features"].array(library="np")
            mask = tr["mask"].array(library="np")
            pT = tr["pT"].array(library="np")
            weight = tr["weight"].array(library="np")
            signal = tr["signal"].array(library="np")
            allPoints = np.concatenate((allPoints,points),axis=0)
            allFeatures = np.concatenate((allFeatures,features),axis=0)
            allMask = np.concatenate((allMask,mask),axis=0)
            allPt = np.concatenate((allPt,pT),axis=0)
            allWeight = np.concatenate((allWeight,weight),axis=0)
            allSignal = np.concatenate((allSignal,signal),axis=0)
            numOfJets = points.shape[0]
            print("Number of jets",numOfJets)
            totalNumOfJets += numOfJets
            allFileLabel += [fileName]*numOfJets
            allMmed += [getPara(fileName,"mMed")]*numOfJets
            allMdark += [getPara(fileName,"mDark")]*numOfJets
            allRinv += [getPara(fileName,"rinv")]*numOfJets
            inputFileIndices += [inputFileIndex]*numOfJets
            inputFileIndex += 1

    # split between train and the others
    trainFraction = sample_fractions[0]
    otherFraction = 1 - trainFraction
    train_indices, other_indices, train_inputFileIndices, other_inputFileIndices = train_test_split(range(totalNumOfJets), inputFileIndices, test_size=otherFraction, random_state=42,stratify=inputFileIndices)
    # split between test and validation
    testFraction = sample_fractions[1]
    testRelativeFraction = testFraction/otherFraction
    test_indices, validation_indices = train_test_split(other_indices, test_size=testRelativeFraction, random_state=42,stratify=other_inputFileIndices)

    print("train_indices",train_indices[:50])
    print("test_indices",test_indices[:50])
    print("validation_indices",validation_indices[:50])
    outputFolder = "processedDataNPZ"
    outputNPZFileName = "processedData"

    # save input features as train, test, and validation sets
    save_npz(train_indices, "train", inputFileNames, allPoints, allFeatures, allMask, allSignal, inputFileIndices, allPt, allWeight, allMmed, allMdark, allRinv, jetConstFeat, outputFolder,outputNPZFileName)
    save_npz(test_indices, "test", inputFileNames, allPoints, allFeatures, allMask, allSignal, inputFileIndices, allPt, allWeight, allMmed, allMdark, allRinv, jetConstFeat, outputFolder,outputNPZFileName)
    save_npz(validation_indices, "validation", inputFileNames, allPoints, allFeatures, allMask, allSignal, inputFileIndices, allPt, allWeight, allMmed, allMdark, allRinv, jetConstFeat, outputFolder,outputNPZFileName)
   
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
    num_classes = args.hyper.num_classes
    num_const = args.features.num_const
    jetConstFeat = args.features.jetConst
    if not os.path.isdir("processedDataNPZ"):
    	os.makedirs("processedDataNPZ")
    process_all_vars(dSet.path, inputFiles, jetConstFeat, num_const, num_classes, sample_fractions)
