import torch
import torch.nn as nn
from torch.nn import functional as f
import torch.utils.data as udata
import os
from models import DNN, DNN_GRF
from dataset import RootDataset, get_sizes
import matplotlib.pyplot as plt
from magiconfig import ArgumentParser, MagiConfigOptions, ArgumentDefaultsRawHelpFormatter
from configs import configs as c
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, roc_auc_score

# ask Kevin how to create training root files for the NN
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def getNNOutput(dataset, model):
    loader = udata.DataLoader(dataset=dataset, batch_size=dataset.__len__(), num_workers=0)
    l, d, p = next(iter(loader))
    labels = l.squeeze(1).numpy()
    data = d.float()
    pT = p.squeeze(1).float().numpy()
    model.eval()
    out, _ = model(data)
    output = f.softmax(out,dim=1)[:,1].detach().numpy()
    return labels, output, pT

def getROCStuff(label, output):
    fpr, tpr, thresholds = roc_curve(label, output)
    auc = roc_auc_score(label, output)
    return fpr, tpr, auc

def getSgBgOutputs(label, output):
    y_Sg = []
    y_Bg = []
    for lt in range(len(label)):
        lbl = label[lt]
        if lbl == 1:
            y_Sg.append(output[lt])
        else:
            y_Bg.append(output[lt])
    return y_Sg, y_Bg

def histplot(data,bins,color,label,alpha=1.0,hatch=None,points=False,facecolorOn=True):
    data,bins = np.histogram(data, bins=bins, density=True)
    bins = np.array([0.5 * (bins[i] + bins[i+1]) for i in range(len(bins)-1)])
    binwidth = bins[1] - bins[0]
    pbins = np.append(bins,bins[-1]+binwidth)
    pdata = np.append(data,data[-1])
    if points:
        pbins = pbins + binwidth/2.
        plt.plot(pbins,pdata,color=color,label=label,marker=".",linestyle="None")
    else:
        plt.step(pbins,pdata,where="post",color=color)
        if facecolorOn:
            facecolor=color
        else:
            facecolor="none"
        plt.fill_between(pbins,pdata, step="post", edgecolor=color, facecolor=facecolor, label=label, alpha=alpha, hatch=hatch)

def main():
    # parse arguments
    parser = ArgumentParser(config_options=MagiConfigOptions(strict = True, default="configs/C1.py"),formatter_class=ArgumentDefaultsRawHelpFormatter)
    parser.add_argument("--outf", type=str, default="logs", help='Name of folder to be used to store outputs')
    parser.add_argument("--model", type=str, default=None, help="Existing model to continue training, if applicable")
    parser.add_config_only(*c.config_schema)
    parser.add_config_only(**c.config_defaults)
    args = parser.parse_args()
    parser.write_config(args, args.outf + "/config_out.py")

    # Choose cpu or gpu
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', args.device)
    if args.device.type == 'cuda':
        gpuIndex = torch.cuda.current_device()
        print("Using GPU named: \"{}\"".format(torch.cuda.get_device_name(gpuIndex)))

    # Load dataset
    print('Loading dataset ...')
    dSet = args.dataset
    sigFiles = dSet.signal
    inputFiles = dSet.background
    hyper = args.hyper
    inputFiles.update(sigFiles)
    varSet = args.features.train
    uniform = args.features.uniform
    dataset = RootDataset(inputFolder=dSet.path, root_file=inputFiles, variables=varSet, uniform=uniform)
    sizes = get_sizes(len(dataset), dSet.sample_fractions)
    train, val, test = udata.random_split(dataset, sizes, generator=torch.Generator().manual_seed(42))
    loader_train = udata.DataLoader(dataset=train, batch_size=hyper.batchSize, num_workers=0)
    loader_val = udata.DataLoader(dataset=val, batch_size=hyper.batchSize, num_workers=0)
    loader_test = udata.DataLoader(dataset=test, batch_size=hyper.batchSize, num_workers=0)
    # # Build model
    model = DNN_GRF(n_var=len(varSet), n_layers=hyper.num_of_layers, n_nodes=hyper.num_of_nodes, n_outputs=2, drop_out_p=hyper.dropout).to(device=args.device)
    print("Loading model from file " + args.model)
    model.load_state_dict(torch.load(args.model))
    model.eval()

    # plot correlation
    df = dataset.dataFrame
    fig, ax = plt.subplots(figsize=(12, 8))
    corr = np.round(df.corr(),2)
    ax = sns.heatmap(corr,cmap="Spectral",annot=True)
    bottom, top = ax.get_ylim()
    left, right = ax.get_xlim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    ax.set_xlim(left - 0.5, right + 0.5)
    plt.savefig(args.outf + "/corrHeatMap.pdf", dpi=fig.dpi)
    fig, ax = plt.subplots(figsize=(12, 8))

    # plot input variable
    print(varSet)
    df["label"] = np.array(dataset.signal)[:,0]
    for var in varSet:
        dataSig = df[var][df["label"] == 1]
        dataBkg = df[var][df["label"] == 0]
        fig, ax = plt.subplots(figsize=(12, 8))
        histplot(dataSig,bins=50,color='xkcd:blue',alpha=0.5,label='Background')
        histplot(dataBkg,bins=50,color='xkcd:red',alpha=1.0,label='Signal',hatch="//",facecolorOn=False)
        ax.set_ylabel('Norm Events')
        ax.set_xlabel(var)
        plt.legend()
        plt.savefig(args.outf + "/{}.pdf".format(var), dpi=fig.dpi)

    # plot ROC curve
    model.to('cpu')
    label_train, output_train, pT_train = getNNOutput(train, model)
    label_test, output_test, pT_test = getNNOutput(test, model)
    fpr_Train, tpr_Train, auc_Train = getROCStuff(label_train, output_train)
    fpr_Test, tpr_Test, auc_Test = getROCStuff(label_test, output_test)
    fig = plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve', pad=45.0)
    plt.plot(fpr_Train, tpr_Train, label="Train (area = {:.3f})".format(auc_Train), color='xkcd:red')
    plt.plot(fpr_Test, tpr_Test, label="Test (area = {:.3f})".format(auc_Test), color='xkcd:black')
    plt.legend(loc='best')
    fig.savefig(args.outf + "/roc_plot.pdf", dpi=fig.dpi)
    plt.close(fig)

    # # plot eff vs pT
    # pTBins = [[0,80],[80,120],[120,170],[170,300],[300,470],[470,600],[600,800],[800,1000],[1000,1400],[1400,1800],[1800,2400],[2400,3200],[3200,np.Inf]]
    pTWidth = 100
    pTBins = np.arange(0,5000,pTWidth)
    pTX = []
    fRate = []
    eff = []
    wpt = 0.5 # NN working point
    for pT in pTBins:
        totSigJet = 0.
        totBkgJet = 0.
        truePos = 0.
        falPos = 0.
        pTRan = [pT,pT+pTWidth]
        if pT == 5000:
            pTRan = [pT,np.Inf]
        pTX.append(np.mean(pTRan))
        for i in range(len(pT_train)):
            pT_t = pT_train[i]
            label_t = label_train[i]
            out_t = output_train[i]
            if pTRan[0] < pT_t < pTRan[1]:
                if label_t == 0:
                    totBkgJet += 1
                    if out_t > wpt:
                        falPos += 1
                else:
                    totSigJet += 1
                    if out_t > wpt:
                        truePos += 1
        if totSigJet == 0:
            eff.append(0)
        else:
            eff.append(truePos/totSigJet)
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.plot(pTX,eff)
    ax.set_ylabel('Efficiency')
    ax.set_xlabel('pT (GeV)')
    plt.savefig(args.outf + "/effVspT.pdf", dpi=fig.dpi)

    # plot discriminator
    bins = np.linspace(0, 1, 100)
    fig, ax = plt.subplots(figsize=(6, 6))
    y_Train_Sg, y_Train_Bg = getSgBgOutputs(label_train, output_train)
    y_test_Sg, y_test_Bg = getSgBgOutputs(label_test, output_test)
    ax.set_title('')
    ax.set_ylabel('Norm Events')
    ax.set_xlabel('Discriminator')
    histplot(y_test_Sg,bins,color='xkcd:red',alpha=1.0,label='Sg Test',hatch="//",facecolorOn=False)
    histplot(y_test_Bg,bins,color='xkcd:blue',alpha=0.5,label='Bg Test')
    histplot(y_Train_Sg, bins, color='xkcd:red',label='Sg Train',points=True)
    histplot(y_Train_Bg, bins, color='xkcd:blue',label='Bg Train',points=True)
    ax.legend(loc='best', frameon=False)
    fig.savefig(args.outf + "/discriminator.pdf", dpi=fig.dpi)

if __name__ == "__main__":
    main()
