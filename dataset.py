import numpy as np
import uproot as up
from magiconfig import ArgumentParser, MagiConfigOptions, ArgumentDefaultsRawHelpFormatter
from configs import configs as c
import torch.utils.data as udata
import torch
import pandas as pd

def getBranch(f,tree,variable,branches,branchList):
    branch = f[tree].pandas.df(variable)
    branch = branch.head(len(branches))
    branchList.append(branch)

def get_all_vars(inputFolder, samples, variables, uniform, mT, weight, tree="tree"):
    dSets = []
    signal = []
    pTs = []
    mTs = []
    weights = []
    for key,fileList in samples.items():
        for fileName in fileList:
            f = up.open(inputFolder  + fileName + ".root")
            branches = f[tree].pandas.df(variables)
            #branches = branches.head(22690) #Hardcoded only taking ~30k events per file while we setup the code; should remove this when we want to do some serious trainings
            #branches = branches.head(5000)
            dSets.append(branches)
            getBranch(f,tree,uniform,branches,pTs)
            getBranch(f,tree,mT,branches,mTs)
            getBranch(f,tree,weight,branches,weights)
            if key == "signal":
                signal += list([0, 1] for _ in range(len(branches)))
            else:
                signal += list([1, 0] for _ in range(len(branches)))
    dataSet = pd.concat(dSets)
    pT = pd.concat(pTs)
    mT = pd.concat(mTs)
    weight = pd.concat(weights)
    return [dataSet,signal,pT,mT,weight]

def get_sizes(l, frac=[0.8, 0.1, 0.1]):
    if sum(frac) != 1.0: raise ValueError("Sum of fractions does not equal 1.0")
    if len(frac) != 3: raise ValueError("Need three numbers in list for train, test, and val respectively")
    train_size = int(frac[0]*l)
    test_size = int(frac[1]*l)
    val_size = l - train_size - test_size
    return [train_size, test_size, val_size]

class RootDataset(udata.Dataset):
    def __init__(self, inputFolder, root_file, variables, uniform, mT, weight):
        dataSet, signal, pT, mTs, weights = get_all_vars(inputFolder, root_file, variables, uniform, mT, weight)
        print(type(dataSet))
        self.root_file = root_file
        self.variables = variables
        self.uniform = uniform
        self.weight = weight
        self.vars = dataSet.astype(float).values
        self.signal = signal
        self.pT = pT.astype(float).values
        self.mTs = mTs.astype(float).values
        self.weights = weights.astype(float).values
        print("Number of events:", len(self.signal))

    #def get_arrays(self):
    #    return np.array(self.signal), torch.from_numpy(self.vars.astype(float).values.copy()).float().squeeze(1)

    def __len__(self):
        return len(self.vars)

    def __getitem__(self, idx):
        data_np = self.vars[idx].copy()
        label_np = np.zeros(1, dtype=np.long).copy()
        pT_np = self.pT[idx].copy()
        mTs_np = self.mTs[idx].copy()
        weights_np = self.weights[idx].copy()

        if self.signal[idx][1]:
            label_np += 1

        data  = torch.from_numpy(data_np)
        label = torch.from_numpy(label_np)
        pT = torch.from_numpy(pT_np).float()
        mTs = torch.from_numpy(mTs_np).float()
        weights = torch.from_numpy(weights_np).float()
        return label, data, pT, mTs, weights

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
    varSet = args.features.train
    print(varSet)
    uniform = args.features.uniform
    mTs = args.features.mT
    weights = args.features.weight
    dataset = RootDataset(dSet.path, inputFiles, varSet, uniform, mTs, weights)
    sizes = get_sizes(len(dataset), [0.8, 0.1, 0.1])
    train, val, test = udata.random_split(dataset, sizes, generator=torch.Generator().manual_seed(42))
    loader = udata.DataLoader(dataset=train, batch_size=train.__len__(), num_workers=0)
    l, d, p, m, w = next(iter(loader))
    labels = l.squeeze(1).numpy()
    data = d.float().numpy()
    pT = p.squeeze(1).float().numpy()
    mTs = m.squeeze(1).float().numpy()
    weights = w.squeeze(1).float().numpy()
    print(labels)
    print(data)
    print(pT)
    print(mTs)
    print(weights)
