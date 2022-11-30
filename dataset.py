import numpy as np
import uproot as up
from magiconfig import ArgumentParser, MagiConfigOptions, ArgumentDefaultsRawHelpFormatter
from configs import configs as c
import torch.utils.data as udata
import torch
import pandas as pd
from tqdm import tqdm

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
        pTs = pData["pT"]
        weights = pData["weight"]
        mMeds = pData["mMed"]
        mDarks = pData["mDark"]
        rinvs = pData["rinv"]
        alphas = pData["alpha"]
        inputFileIndex = pData["inputFileIndices"]
        inputFileNames = pData["inputFileNames"]
        inputFeaturesVarName = pData["inputFeaturesVarName"]
        # self.vars = dataSet.astype(float).values
        self.points = inputPoints
        self.features = inputFeatures
        self.jetFeatures = inputJetFeatures
        self.signal = signal
        self.pTs = pTs
        self.weights = weights
        self.mMeds = mMeds
        self.mDarks = mDarks
        self.rinvs = rinvs
        self.alphas = alphas
        self.inputFileIndex = inputFileIndex
        self.inputFileNames = inputFileNames
        self.inputFeaturesVarName = inputFeaturesVarName
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
        pTs_np = np.array([self.pTs[idx]]).copy()
        weights_np = np.array([self.weights[idx]]).copy()
        mMeds_np = np.array([self.mMeds[idx]]).copy()
        mDarks_np = np.array([self.mDarks[idx]]).copy()
        rinvs_np = np.array([self.rinvs[idx]]).copy()
        alphas_np = np.array([np.compat.long(self.alphas[idx])]).copy()
        inputFileIndex_np = np.array([np.compat.long(self.inputFileIndex[idx])]).copy()
        label_np += np.where(self.signal[idx] == 1)[0][0] #Calculates label 
        
        points  = torch.from_numpy(points_np)
        features  = torch.from_numpy(features_np)
        jetFeatures  = torch.from_numpy(jetFeatures_np)
        # print("Data inside getitem")
        # print(data)
        label = torch.from_numpy(label_np)
        inputFileIndex = torch.from_numpy(inputFileIndex_np)
        pTs = torch.from_numpy(pTs_np).float()
        weights = torch.from_numpy(weights_np).float()
        mMeds = torch.from_numpy(mMeds_np).float()
        mDarks = torch.from_numpy(mDarks_np).float()
        rinvs = torch.from_numpy(rinvs_np).float()
        alphas = torch.from_numpy(alphas_np)

        return label, points, features, jetFeatures, inputFileIndex, pTs, weights, mMeds, mDarks, rinvs, alphas

def getCondition(inputFileNames,key,mcT):
    conditions = np.zeros(len(mcT),dtype=bool)
    for i in range(len(inputFileNames)):
        fileName = inputFileNames[i]
        if key in fileName:
            conditions = conditions | (mcT == i)
    return conditions

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
    weights = args.features.weight
    numConst = args.hyper.numConst
    dataset = RootDataset("processedDataNPZ/processedData_nc100_train_pTWeightedByBinAndProportion.npz")
    print("inputFileNames:",dataset.inputFileNames)
    inputFileNames = dataset.inputFileNames
    #sizes = get_sizes(len(dataset), dSet.sample_fractions)
    #train, val, test = udata.random_split(dataset, sizes, generator=torch.Generator().manual_seed(1000))
    #print("train.__len__()",train.__len__())
    #print(train)
    loader = udata.DataLoader(dataset=dataset, shuffle=True, batch_size=dataset.__len__(), num_workers=0, generator=torch.Generator().manual_seed(1000)) # generator=torch.Generator().manual_seed(1000)
    l, po, fea, jfea, inp, p, w, med, dark, rinv, alpha = next(iter(loader))
    labels = l.squeeze(1).numpy()
    inputFileIndex = inp.squeeze(1).numpy()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    points = po.float().numpy()
    trainPoints = po.float().to(device)
    features = fea.float().numpy()
    trainFeatures = fea.float().to(device)
    pTs = p.squeeze(1).float().numpy()
    weights = w.squeeze(1).float().numpy()
    meds = med.squeeze(1).float().numpy()
    darks = dark.squeeze(1).float().numpy()
    rinvs = rinv.squeeze(1).float().numpy()
    alphas = alpha.squeeze(1).numpy()
    print("QCD",np.sum(getCondition(inputFileNames,"QCD",inputFileIndex)) )
    print("tree_QCD_Pt_600to800_PN",np.sum(getCondition(inputFileNames,"tree_QCD_Pt_600to800_PN",inputFileIndex)) )
    print("tree_SVJ_mMed-2000_mDark-100_rinv-0p3_alpha-peak_yukawa-1_PN",np.sum(getCondition(inputFileNames,"tree_SVJ_mMed-2000_mDark-100_rinv-0p3_alpha-peak_yukawa-1_PN",inputFileIndex)) )
    print("labels first 20:", list(labels[:20]))
    print("labels last 20:", list(labels[-20:]))
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
    print("inputFileIndex:", inputFileIndex)
    unique, unique_counts = np.unique(inputFileIndex,return_counts=True)
    for i in range(len(unique)):
        print(inputFileNames[unique[i]],unique_counts[i])
    print("pT:", pTs)
    print("pT min:", np.amin(pTs))
    print("pT max:", np.amax(pTs))
    print("weights", weights)
    print("meds:", np.unique(meds))
    print("darks:", np.unique(darks))
    print("rinvs:", np.unique(rinvs))
    print("alphas", np.unique(alphas))
