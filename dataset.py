import numpy as np
import uproot as up
import matplotlib.pyplot as plt
import matplotlib as mpl
import random 
import torch.utils.data as udata
import torch 
import math

def get_all_vars(file_path, variables, tree="tree"):
    f = up.open(file_path)
    branches = f[tree].pandas.df(variables)
    return branches

class RootDataset(udata.Dataset):
    def __init__(self, root_file, variables):
        self.root_file = root_file
        self.variables = variables
        self.vars = get_all_vars(root_file, variables)
        self.signal = False

    def __len__(self):
        return len(self.vars)

    def __getitem__(self, idx):
        data_np = self.vars.astype(float).values[idx]
        label_np = torch.zeros(1, dtype=torch.long)
        if self.signal: label_np += 1
                
        label = label_np
        data = torch.from_numpy(data_np.copy())
        return label, data 

if __name__=="__main__":
    dataset = RootDataset("tree_QCD_Pt_600to800_MC2017.root", ["pt","eta"])
    
    for i in range(10):
        print("---"*50)
        label, data = dataset.__getitem__(i)         

