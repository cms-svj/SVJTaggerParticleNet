import numpy as np
import uproot as up
import matplotlib.pyplot as plt
import matplotlib as mpl
from magiconfig import ArgumentParser, MagiConfigOptions, ArgumentDefaultsRawHelpFormatter
from configs import configs as c
import random
import torch.utils.data as udata
import torch
import math
import pandas as pd

def get_all_vars(samples, variables, tree="tree"):
    dSets = []
    signal = []
    for fileName in samples:
        f = up.open(fileName)
        branches = f[tree].pandas.df(variables)
        branches = branches.head(22690) #Hardcoded only taking 30k events per file while we setup the code; should remove this when we want to do some serious trainings
        dSets.append(branches)
        if "SVJ" in fileName:
            signal += list([0, 1] for _ in range(len(branches)))
        else:
            signal += list([1, 0] for _ in range(len(branches)))
    dataSet = pd.concat(dSets)
    return [dataSet,signal]

class RootDataset(udata.Dataset):
    def __init__(self, root_file, variables,signal=False):
        dataInfo = get_all_vars(root_file, variables)
        self.root_file = root_file
        self.variables = variables        
        self.vars = dataInfo[0]
        self.signal = dataInfo[1]

    def get_arrays(self):
        return np.array(self.signal), torch.from_numpy(self.vars.astype(float).values.copy()).float().squeeze(1)

    def __len__(self):
        return len(self.vars)

    def __getitem__(self, idx):
        data_np = self.vars.astype(float).values[idx]
        label = torch.zeros(1, dtype=torch.long)
        if self.signal[idx][1]: label += 1
        data = torch.from_numpy(data_np.copy())
        return label, data

if __name__=="__main__":
    # parse arguments
    parser = ArgumentParser(config_options=MagiConfigOptions(strict = True),formatter_class=ArgumentDefaultsRawHelpFormatter)
    parser.add_config_only(*c.config_schema)
    parser.add_config_only(**c.config_defaults)
    args = parser.parse_args()
    inputFiles = []
    dSet = args.dataset
    for bkg,fileList in dSet.background.items():
        inputFiles += [dSet.path + fileName + '.root' for fileName in fileList]
    for sig,fileList in dSet.signal.items():
        inputFiles += [dSet.path + fileName + '.root' for fileName in fileList]
    print(inputFiles)
    varSet = args.features.train
    print(varSet)
    dataset = RootDataset(inputFiles, varSet, signal=False)

    for i in range(10):
        print("---"*50)
        label, data = dataset.__getitem__(i)
        print(label,data)

