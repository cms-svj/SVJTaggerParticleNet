import numpy as np
import mplhep as hep
import sys

labelDict = {
    0:"QCD",
    1:"TTJets",
    2:"Dark",
}

def getCondition(label,key):
    return label == key

def countJets(labelDict, label, key):
    print("Jet {}: {}".format(labelDict[key],np.sum(getCondition(label,key))) )

def getReferencePtHist(label,pT,weight,randomGenerator):
    pTBin = np.arange(0,3000,50)
    useLab = 2
    fracToUse = 1.0
    indices = np.where(getCondition(label,useLab))[0]
    print("indices length:", len(indices))
    useInd = randomGenerator.choice(indices,int(len(indices)*fracToUse),replace=False)
    print("useInd length:",len(useInd))
    referencePtHist, bins = np.histogram(np.take(pT,useInd,axis=0),pTBin)
    return referencePtHist,pTBin,useInd

def getSigJetIndices(referenceHist,pTBin,label,key,pT):
    digitpT = np.digitize(pT,pTBin) - 1
    jetCond = getCondition(label,key)
    finalIndices = []
    for j in range(len(referenceHist)):
        nJ_j = referenceHist[j]
        allJ_j_pos = np.where((digitpT == j) & (jetCond))[0]
        if (nJ_j == 0) or (len(allJ_j_pos) == 0):
            continue
        if len(allJ_j_pos) > nJ_j:
            finalIndices += list(randomGenerator.choice(allJ_j_pos,nJ_j,replace=False))
        else:
            finalIndices += list(randomGenerator.choice(allJ_j_pos,nJ_j,replace=True))
    return finalIndices

# this not only samples the background jets in a way that match some reference histogram, it also makes sure that the proportion of jets in each pT bin matches the actual proportion of jets from each subsample in that bin
def getBkgJetIndices(referenceHist,pTBin,key,inputFileNames,inputFileIndices,pT,weight,randomGenerator):
    weightedHistList = []
    histFileConList = []
    finalIndices = []
    for i in range(len(inputFileNames)):
        inFile = inputFileNames[i]
        if key in inFile:
            cond = inputFileIndices == i
            weightedHist, bins = np.histogram(pT[cond],pTBin,weights=weight[cond])
            weightedHistList.append(weightedHist)
            histFileConList.append(cond)
    totalWeightedHist = np.sum(weightedHistList,axis=0)
    numberOfJetsToKeepList = []
    for weightedHist in weightedHistList:
        numberOfJetsToKeepList.append((weightedHist/totalWeightedHist) * referenceHist)
    digitpT = np.digitize(pT,pTBin) - 1
    for i in range(len(histFileConList)):
        histFileCon = histFileConList[i]
        numberOfJetsToKeep = numberOfJetsToKeepList[i]
        print(inputFileNames[i])
        for j in range(len(numberOfJetsToKeep)):
            nJ_j = int(np.nan_to_num(numberOfJetsToKeep[j],posinf=0, neginf=0))
            allJ_j_pos = np.where((digitpT == j) & (histFileCon))[0]
            if (nJ_j == 0) or (len(allJ_j_pos) == 0):
                continue
            if len(allJ_j_pos) > nJ_j:
                finalIndices += list(randomGenerator.choice(allJ_j_pos,nJ_j,replace=False))
            else:
                finalIndices += list(randomGenerator.choice(allJ_j_pos,nJ_j,replace=True))
    return finalIndices

trainType = sys.argv[1] # (train or validation)
inFolder = "processedDataNPZ"
inNPZFileName = "processedData_{}".format(trainType)
npzFile = np.load("{}/{}.npz".format(inFolder,inNPZFileName))
inputPoints = npzFile["inputPoints"]
inputFeatures = npzFile["inputFeatures"]
masks = npzFile["masks"]
inputFileIndices = npzFile["inputFileIndices"]
signal = npzFile["signal"]
pT = npzFile["pT"]
weight = npzFile["weight"]
mMed = npzFile["mMed"]
mDark = npzFile["mDark"]
rinv = npzFile["rinv"]
label = np.where(signal==1)[1]
inputFileNames = npzFile["inputFileNames"]
inputFeaturesVarName = npzFile["inputFeaturesVarName"]
print("Before weighting:")
print("Total jets used in training",len(inputFileIndices))
for key in np.unique(label):
    countJets(labelDict, label, key)

randomGenerator = np.random.default_rng(888)
# reference histogram is the sum of all SVJ signals' pT
referenceHist,pTBin,referenceIndices = getReferencePtHist(label,pT,weight,randomGenerator)
QCDFinalIndices = getBkgJetIndices(referenceHist,pTBin,"QCD",inputFileNames,inputFileIndices,pT,weight,randomGenerator)
TTJetsFinalIndices = getBkgJetIndices(referenceHist,pTBin,"TTJets",inputFileNames,inputFileIndices,pT,weight,randomGenerator)
#mixDarkIndices = getSigJetIndices(referenceHist,pTBin,label,3,pT)
finalIndices = QCDFinalIndices + TTJetsFinalIndices + list(referenceIndices) # + mixDarkIndices 

useLabel = np.take(label,finalIndices,axis=0)
print("Number of jets by jet category")
print(np.unique(useLabel,return_counts=True))

#raise Exception("Arretez")

dictOut = {
    "inputPoints":np.take(inputPoints,finalIndices,axis=0),
    "inputFeatures":np.take(inputFeatures,finalIndices,axis=0),
    "masks":np.take(masks,finalIndices,axis=0),
    "inputFileIndices":np.take(inputFileIndices,finalIndices,axis=0),
    "signal":np.take(signal,finalIndices,axis=0),
    "pT":np.take(pT,finalIndices,axis=0),
    "weight":np.take(weight,finalIndices,axis=0),
    "mMed":np.take(mMed,finalIndices,axis=0),
    "mDark":np.take(mDark,finalIndices,axis=0),
    "rinv":np.take(rinv,finalIndices,axis=0),
    "inputFileNames":inputFileNames,
    "inputFeaturesVarName":inputFeaturesVarName
}

#print("pT",len(dictOut['pT']))
#print("weight",len(dictOut['weight']))
#print("newweight",len(dictOut['newweight']))
#print("signal",len(dictOut['signal']))
#print("inputFileIndices",len(dictOut['inputFileIndices']))    
#print("inputFileNames",inputFileNames)
#print("inputPoints",dictOut['inputPoints'].shape)
#print("inputFeatures",dictOut['inputFeatures'].shape)
#print("inputJetFeatures",dictOut['inputJetFeatures'].shape)        
np.savez_compressed("{}/{}_uniformPt.npz".format(inFolder,inNPZFileName),**dictOut)
