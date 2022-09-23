import os
import numpy as np
from collections import OrderedDict
import itertools
import re

def strInd(data,key):
    ind = 0
    for i in range(len(data)):
        if key in data[i]:
            ind = i
    return ind

def binToDec(numStr):
    dec = 0
    for i in range(len(numStr)):
        dec += int(numStr[i])*(2**(len(numStr)-i-1))
    return dec

def replaceSubstrs(strg,dictRep):
    s = strg
    for key,item in dictRep.items():
        s = s.replace(key,item)
    return s

# vary learning rate on our own [0.001,0.005,0.01]
batchDict = OrderedDict([
    ("learning_rate",[0.001]),
    ("batchSize", [512]),
    ("numConst", [100]),
    ("num_of_k_nearest", [16]),
    ("num_of_edgeConv_dim", [[64,128],[64,128,256]]),
    ("num_of_edgeConv_convLayers", [2,3]),
    ("num_of_fc_layers", [5,7]),
    ("num_of_fc_nodes", [256]),
    ("fc_dropout", [0.3]),
    ("epochs",[80,120]),
    ("lambdaDC",[0.0]),
    ("rseed", [30]),
])

shortDict = {
    "learning_rate": "lr",
    "batchSize": "bs",
    "numConst": "nc",
    "num_of_k_nearest": "nk",
    "num_of_edgeConv_dim": "ed",
    "num_of_edgeConv_convLayers": "el",
    "num_of_fc_layers": "fl",
    "num_of_fc_nodes": "fn",
    "fc_dropout": "fd",
    "epochs": "ep",
    "lambdaDC": "dc",
    "rseed": "rs",
}

with open('C_tch_jConst.py', 'r') as file:
    data = file.readlines()

trainingList = []
locList = []
keyList = []
trainLabel = ""
extraComment = "_newSetupTTQCD"
for key,item in batchDict.items():
    if len(item) > 0:
        keyList.append(key)
        locList.append(strInd(data,key))
        trainingList.append(item)
        trainLabel = "1" + trainLabel
    else:
        trainLabel = "0" + trainLabel
allTrainings = list(itertools.product(*trainingList))

dictRep = {
    ".":"p",
    " ":"",
    "[":"",
    "]":"",
    ",":"_"
}

count = 0
for training in allTrainings:
    outDir = ""
    for i in range(len(locList)):
        loc = locList[i]
        trPara = training[i]
        key = keyList[i]
        data[loc] = 'config.hyper.{} = {}\n'.format(key,trPara)
        outDir += "{}{}_".format(shortDict[key],replaceSubstrs(str(trPara),dictRep))
    outDir += str(binToDec(trainLabel))
    outDir += extraComment
    # # training setup
    count += 1
    with open('batch/E_{}_{}.py'.format(count,outDir), 'w+') as file:
        file.writelines( data )
    # os.system("python train.py --outf logs/{} -C configs/C_tch.py".format(outDir))
    # # # validation only
    # os.system("python validation.py --model net.pth --outf logs/{0} -C logs/{0}/config_out.py".format(outDir))

# tch: python train.py --outf logs/{} -C configs/C_tch.py
# testDarkOnly_tchannel_upto3000_allSig
# python validation.py --model net.pth --outf logs/test_tch_normMeanStd_lr0p0001_bs2000_nf2_nt4_nn40_ep20 -C logs/test_tch_normMeanStd_lr0p0001_bs2000_nf2_nt4_nn40_ep20/config_out.py
