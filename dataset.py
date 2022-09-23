import numpy as np
import uproot as up
from magiconfig import ArgumentParser, MagiConfigOptions, ArgumentDefaultsRawHelpFormatter
from configs import configs as c
import torch.utils.data as udata
import torch
import pandas as pd
from tqdm import tqdm

def splitArrayByChunkSize(alist,chunkSize):
    if chunkSize > len(alist):
        chunkSize = len(alist)
    numOfChunks = len(alist) // chunkSize
    aListChunks = []
    start = 0
    for i in range(numOfChunks):
        end = start + chunkSize
        aListChunks.append(alist[start:end])
        start = end
    return aListChunks

def splitDataSetEvenly(dataset,rng,numOfEpoch=10):
    inputFileIndex = dataset.inputFileIndex
    signalFileIndex = dataset.signalFileIndex
    listIndices = np.arange(len(inputFileIndex))
    inputFileIndex = np.array(inputFileIndex)
    minOcc = np.amin(np.unique(inputFileIndex,return_counts=True)[1])
    uniqueFileIndices = np.unique(inputFileIndex)
    numOfSigFiles = len(signalFileIndex)
    numOfBkgFiles = len(uniqueFileIndices) - numOfSigFiles
    allSamplesIndices = []
    numOfSets = []
    for i in uniqueFileIndices:
        chunkSize = minOcc
        subIndexList = listIndices[inputFileIndex==i]
        rng.shuffle(subIndexList)
        if i not in signalFileIndex:
            chunkSize = int(minOcc*(numOfSigFiles/numOfBkgFiles)) # this is assuming there are more background events than signal events in general
        indexSet = splitArrayByChunkSize(subIndexList,chunkSize)
        allSamplesIndices.append(indexSet)
        numOfSets.append(len(indexSet))
    randBalancedSet = []
    for i in range(numOfEpoch):
        randomIndexSet = [rng.randint(0,high=ind) for ind in numOfSets]
        indicesForEpoch = np.array([],dtype=int)
        for j in range(len(allSamplesIndices)):
            indicesForEpoch = np.concatenate((indicesForEpoch,allSamplesIndices[j][randomIndexSet[j]]))
        randBalancedSet.append(indicesForEpoch)
    return randBalancedSet

def get_sizes(l, frac=[0.8, 0.1, 0.1]):
    if sum(frac) != 1.0: raise ValueError("Sum of fractions does not equal 1.0")
    if len(frac) != 3: raise ValueError("Need three numbers in list for train, test, and val respectively")
    train_size = int(frac[0]*l)
    test_size = int(frac[1]*l)
    val_size = l - train_size - test_size
    return [train_size, test_size, val_size]

class RootDataset(udata.Dataset):
    def __init__(self, processedFile):
        pData = np.load(processedFile)
        inputPoints = pData["inputPoints"]
        inputFeatures = pData["inputFeatures"]
        inputJetFeatures = pData["inputJetFeatures"]
        signal = pData["signal"]
        mcType = pData["mcType"]
        pTLab = pData["pTLab"]
        pTs = pData["pT"]
        mTs = pData["mT"]
        weights = pData["weight"]
        mMeds = pData["mMed"]
        mDarks = pData["mDark"]
        rinvs = pData["rinv"]
        alphas = pData["alpha"]
        inputFileIndex = pData["inputFileIndices"]
        signalFileIndex = pData["signalFileIndex"]
        # self.vars = dataSet.astype(float).values
        self.points = inputPoints
        self.features = inputFeatures
        self.jetFeatures = inputJetFeatures
        self.signal = signal
        self.mcType = mcType
        self.pTLab = pTLab
        self.pTs = pTs
        self.mTs = mTs
        self.weights = weights
        self.mMeds = mMeds
        self.mDarks = mDarks
        self.rinvs = rinvs
        self.alphas = alphas
        self.inputFileIndex = np.array(inputFileIndex)
        self.signalFileIndex = np.array(signalFileIndex)
        print("Number of events:", len(self.signal))

    #def get_arrays(self):
    #    return np.array(self.signal), torch.from_numpy(self.vars.astype(float).values.copy()).float().squeeze(1)

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        points_np = self.points[idx].copy()
        features_np = self.features[idx].copy()
        jetFeatures_np = self.jetFeatures[idx].copy()
        label_np = np.zeros(1, dtype=np.compat.long).copy()
        mcType_np = np.array([np.compat.long(self.mcType[idx])]).copy()
        pTLab_np = np.array([np.compat.long(self.pTLab[idx])]).copy()
        pTs_np = np.array([self.pTs[idx]]).copy()
        mTs_np = self.mTs[idx].copy()
        weights_np = np.array([self.weights[idx]]).copy()
        mMeds_np = np.array([self.mMeds[idx]]).copy()
        mDarks_np = np.array([self.mDarks[idx]]).copy()
        rinvs_np = np.array([self.rinvs[idx]]).copy()
        alphas_np = np.array([np.compat.long(self.alphas[idx])]).copy()

        if self.signal[idx][1]:
            label_np += 1

        points  = torch.from_numpy(points_np)
        features  = torch.from_numpy(features_np)
        jetFeatures  = torch.from_numpy(jetFeatures_np)
        # print("Data inside getitem")
        # print(data)
        label = torch.from_numpy(label_np)
        mcType = torch.from_numpy(mcType_np)
        pTLab = torch.from_numpy(pTLab_np)
        pTs = torch.from_numpy(pTs_np).float()
        mTs = torch.from_numpy(mTs_np).float()
        weights = torch.from_numpy(weights_np).float()
        mMeds = torch.from_numpy(mMeds_np).float()
        mDarks = torch.from_numpy(mDarks_np).float()
        rinvs = torch.from_numpy(rinvs_np).float()
        alphas = torch.from_numpy(alphas_np)
        return label, points, features, jetFeatures, mcType, pTLab, pTs, mTs, weights, mMeds, mDarks, rinvs, alphas

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
    print(inputFiles)
    varSetjetConst = args.features.jetConst
    inputFeatureVars = [var for var in varSetjetConst if var not in ["jCsthvCategory","jCstEvtNum","jCstJNum"]]
    print("Input jet constituent features:",inputFeatureVars)
    varSetjetVariables = args.features.jetVariables
    print("Input jet features:",varSetjetVariables)
    pTBins = args.hyper.pTBins
    print(pTBins)
    uniform = args.features.uniform
    mTs = args.features.mT
    weights = args.features.weight
    numConst = args.hyper.numConst
    dataset = RootDataset("processedDataNPZ/processedData_nc{}.npz".format(numConst))
    sizes = get_sizes(len(dataset), dSet.sample_fractions)
    train, val, test = udata.random_split(dataset, sizes, generator=torch.Generator().manual_seed(1000))
    print("train.__len__()",train.__len__())
    loader = udata.DataLoader(dataset=train, batch_size=train.__len__(), num_workers=0)
    l, po, fea, jfea, mct, pl, p, m, w, med, dark, rinv, alpha = next(iter(loader))
    labels = l.squeeze(1).numpy()
    mcType = mct.squeeze(1).numpy()
    pTLab = pl.squeeze(1).numpy()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    points = po.float().numpy()
    trainPoints = po.float().to(device)
    features = fea.float().numpy()
    trainFeatures = fea.float().to(device)
    pTs = p.squeeze(1).float().numpy()
    mTs = m.squeeze(1).float().numpy()
    weights = w.squeeze(1).float().numpy()
    meds = med.squeeze(1).float().numpy()
    darks = dark.squeeze(1).float().numpy()
    rinvs = rinv.squeeze(1).float().numpy()
    alphas = alpha.squeeze(1).numpy()
    print("labels:", labels)
    print("Number of signals: ",len(labels[labels==1]))
    print("Number of backgrounds: ",len(labels[labels==0]))
    print("Information for Points:")
    print("Input has nan:",np.isnan(np.sum(points)))
    print("inputMean:", np.mean(points,axis=0))
    print("inputSTD:", np.std(points,axis=0))
    print("Information for Features:")
    print("Input has nan:",np.isnan(np.sum(features)))
    print("inputMean:", np.mean(features,axis=0))
    print("inputSTD:", np.std(features,axis=0))
    print("mcType:", mcType)
    print("pTLab:", pTLab)
    print("mT:", mTs)
    print("pT:", pTs)
    print("pT min:", np.amin(pTs))
    print("pT max:", np.amax(pTs))
    print("weights", weights)
    print("meds:", np.unique(meds))
    print("darks:", np.unique(darks))
    print("rinvs:", np.unique(rinvs))
    print("alphas", np.unique(alphas))
