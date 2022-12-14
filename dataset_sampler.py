import numpy as np
import mplhep as hep

def getCondition(inputFileNames,key,inputFileIndices):
    conditions = np.zeros(len(inputFileIndices),dtype=bool)
    for i in range(len(inputFileNames)):
        fileName = inputFileNames[i]
        if key in fileName:
            conditions = conditions | (inputFileIndices == i)
    return conditions

def countJets(inputFileNames,key,inputFileIndices):
    print("{}: {}".format(key,np.sum(getCondition(inputFileNames,key,inputFileIndices))) )

# we are matching the pT distributions to the SVJPt distribution
def getReferencePtHist(inputFileNames,inputFileIndices,pT,weight):
    pTBin = np.arange(0,3000,50)
    SVJCond = getCondition(inputFileNames,"SVJ",inputFileIndices)
    referencePtHist, bins = np.histogram(pT[SVJCond],pTBin)
    return referencePtHist,pTBin

def getJetIndices(referenceHist,pTBin,key,inputFileNames,inputFileIndices,pT,weight):
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
                finalIndices += list(np.random.choice(allJ_j_pos,nJ_j,replace=False))
            else:
                finalIndices += list(np.random.choice(allJ_j_pos,nJ_j,replace=True))
    return finalIndices

inFolder = "processedDataNPZ"
inNPZFileName = "processedData_nc100_train"
npzFile = np.load("{}/{}.npz".format(inFolder,inNPZFileName))
inputPoints = npzFile["inputPoints"]
inputFeatures = npzFile["inputFeatures"]
inputJetFeatures = npzFile["inputJetFeatures"]
signal = npzFile["signal"]
pT = npzFile["pT"]
weight = npzFile["weight"]
mMed = npzFile["mMed"]
mDark = npzFile["mDark"]
rinv = npzFile["rinv"]
alpha = npzFile["alpha"]
inputFileNames = npzFile["inputFileNames"]
inputFileIndices = npzFile["inputFileIndices"]
inputFeaturesVarName = npzFile["inputFeaturesVarName"]
print("Before weighting:")
print("Total jets used in training",len(inputFileIndices))
countJets(inputFileNames,"QCD",inputFileIndices)
countJets(inputFileNames,"TTJets",inputFileIndices)
countJets(inputFileNames,"SVJ",inputFileIndices)
# reference histogram is the sum of all SVJ signals' pT
referenceHist,pTBin = getReferencePtHist(inputFileNames,inputFileIndices,pT,weight)
QCDFinalIndices = getJetIndices(referenceHist,pTBin,"QCD",inputFileNames,inputFileIndices,pT,weight)
TTJetsFinalIndices = getJetIndices(referenceHist,pTBin,"TTJets",inputFileNames,inputFileIndices,pT,weight)
SVJCond = getCondition(inputFileNames,"SVJ",inputFileIndices)
SVJFinalIndices = np.where(SVJCond)[0]
finalIndices = QCDFinalIndices + TTJetsFinalIndices + list(SVJFinalIndices)
print("QCD jets used in training",len(QCDFinalIndices))
print("TTJets jets used in training",len(TTJetsFinalIndices))
print("SVJ jets used in training",len(SVJFinalIndices))
print("Total jets used in training",len(finalIndices))

dictOut = {
    "inputPoints":np.take(inputPoints,finalIndices,axis=0),
    "inputFeatures":np.take(inputFeatures,finalIndices,axis=0),
    "inputJetFeatures":np.take(inputJetFeatures,finalIndices,axis=0),
    "inputFileIndices":np.take(inputFileIndices,finalIndices,axis=0),
    "signal":np.take(signal,finalIndices,axis=0),
    "pT":np.take(pT,finalIndices,axis=0),
    "weight":np.take(weight,finalIndices,axis=0),
    "mMed":np.take(mMed,finalIndices,axis=0),
    "mDark":np.take(mDark,finalIndices,axis=0),
    "rinv":np.take(rinv,finalIndices,axis=0),
    "alpha":np.take(alpha,finalIndices,axis=0),
    "inputFileNames":inputFileNames,
    "inputFeaturesVarName":inputFeaturesVarName
}

countJets(inputFileNames,"QCD",dictOut["inputFileIndices"])
countJets(inputFileNames,"TTJets",dictOut["inputFileIndices"])
countJets(inputFileNames,"SVJ",dictOut["inputFileIndices"])

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
