import numpy as np
import uproot as up
from magiconfig import ArgumentParser, MagiConfigOptions, ArgumentDefaultsRawHelpFormatter
from configs import configs as c
import torch.utils.data as udata
import torch
import pandas as pd

def get_all_vars(inputFolder, samples, variables, tree="tree"):
    dSets = []
    signal = []
    for key,fileList in samples.items():
        for fileName in fileList:
            f = up.open(inputFolder  + fileName + ".root")
            branches = f[tree].pandas.df(variables)
            branches = branches.head(22690) #Hardcoded only taking ~30k events per file while we setup the code; should remove this when we want to do some serious trainings
            # branches = branches.head(5000)
            dSets.append(branches)
            if key == "signal":
                signal += list([0, 1] for _ in range(len(branches)))
            else:
                signal += list([1, 0] for _ in range(len(branches)))
    dataSet = pd.concat(dSets)
    return [dataSet,signal]

def get_sizes(l, frac=[0.8, 0.1, 0.1]):
    if sum(frac) != 1.0: raise ValueError("Sum of fractions does not equal 1.0")
    if len(frac) != 3: raise ValueError("Need three numbers in list for train, test, and val respectively")
    train_size = int(frac[0]*l)
    test_size = int(frac[1]*l)
    val_size = l - train_size - test_size
    return [train_size, test_size, val_size]

class RootDataset(udata.Dataset):
    def __init__(self, inputFolder, root_file, variables):
        dataInfo = get_all_vars(inputFolder, root_file, variables)
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
    sigFiles = dSet.signal
    inputFiles = dSet.background
    inputFiles.update(sigFiles)
    print(inputFiles)
    varSet = args.features.train
    print(varSet)
    dataset = RootDataset(dSet.path, inputFiles, varSet)

    #for i in range(10):
    #    print("---"*50)
    #    label, data = dataset.__getitem__(i)
    #    print(label,data)

    sizes = get_sizes(len(dataset), [0.8, 0.1, 0.1])
    train, val, test = udata.random_split(dataset, sizes, generator=torch.Generator().manual_seed(42))

    print("-"*50)
    print(dataset.variables)
    print(train.variables)

    loader_train = udata.DataLoader(dataset=train, batch_size=5000, num_workers=0)
    loader_val = udata.DataLoader(dataset=val, batch_size=5000, num_workers=0)
    loader_test = udata.DataLoader(dataset=test, batch_size=5000, num_workers=0)
