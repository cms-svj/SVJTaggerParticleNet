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
mplColors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

def getNNOutput(dataset, model):
    loader = udata.DataLoader(dataset=dataset, batch_size=dataset.__len__(), num_workers=0)
    l, d, pl, p, m, w = next(iter(loader))
    labels = l.squeeze(1).numpy()
    data = d.float()
    # print(p)
    pTL = pl.squeeze(1).float().numpy()
    pT = p.squeeze(1).float().numpy()
    mT = m.squeeze(1).float().numpy()
    weight = w.squeeze(1).float().numpy()
    model.eval()
    out_tag, out_pTClass = model(data)
    input = data.squeeze(1).numpy()
    # output_pTClass = out_pTClass[:,0].detach().numpy()
    output_pTClass = f.softmax(out_pTClass,dim=1).detach().numpy()
    output_tag = f.softmax(out_tag,dim=1)[:,1].detach().numpy()
    # print(output_pTClass)
    # print(output_tag)
    # raise ValueError('Trying to stop the code here.')
    return labels, input, output_tag, output_pTClass, pTL, pT, mT, weight

def getROCStuff(label, output, weights=None):
    fpr, tpr, thresholds = roc_curve(label, output, sample_weight=weights)
    auc = roc_auc_score(label, output)
    return fpr, tpr, auc

def getSgBgOutputs(label, output, weights):
    sigCond = label==1
    bkgCond = np.logical_not(sigCond)
    y_Sg = output[sigCond]
    y_Bg = output[bkgCond]
    w_Sg = weights[sigCond]
    w_Bg = weights[bkgCond]

    return y_Sg, y_Bg, w_Sg, w_Bg

def histMake(data,bins,weights=None,norm=True):
    data,bins = np.histogram(data, bins=bins, weights=weights, density=norm)
    bins = np.array([0.5 * (bins[i] + bins[i+1]) for i in range(len(bins)-1)])
    binwidth = bins[1] - bins[0]
    pbins = np.append(bins,bins[-1]+binwidth)
    pdata = np.append(data,data[-1])
    return np.array(pbins),np.array(pdata)

def histplot(pdata,pbins,color,label,alpha=1.0,hatch=None,points=False,facecolorOn=True,ax=plt):
    if points:
        ax.plot(pbins[:-1],pdata[:-1],color=color,label=label,marker=".",linestyle="None")
    else:
        ax.step(pbins,pdata,where="post",color=color)
        if facecolorOn:
            facecolor=color
        else:
            facecolor="none"
        ax.fill_between(pbins,pdata, step="post", edgecolor=color, facecolor=facecolor, label=label, alpha=alpha, hatch=hatch)

def histMakePlot(data,bins,color,label,weights=None,alpha=1.0,hatch=None,points=False,facecolorOn=True,norm=True,ax=plt):
    pbins,pdata = histMake(data,bins,weights=weights,norm=norm)
    histplot(pdata,pbins,color,label,alpha=alpha,hatch=hatch,points=points,facecolorOn=facecolorOn,ax=ax)

def plotEffvsVar(binX,var_train,w_train,label_train,output_train_tag,varLabel,outf,wpt=0.5,sig=True):
    eff = []
    binWidth = binX[1]-binX[0]
    print(varLabel)
    for j in range(len(binX)):
        binVal = binX[j]
        if j < len(binX)-1:
            varCond = np.absolute(binVal-var_train) < binWidth/2.
        else:
            varCond = var_train > binVal - binWidth/2.
        output_pTC = output_train_tag[varCond]
        label_pTC = label_train[varCond]
        weights_pTC = w_train[varCond]
        output_pTC_wpt = np.where(output_pTC>wpt,1,0)
        if sig:
            ylabel = "Signal Efficiency"
            plotname = outf + "/sigEffVs{}.pdf".format(varLabel)
            totalCount = label_pTC
        else:
            ylabel = "Mistag Rate"
            plotname = outf + "/mistagVs{}.pdf".format(varLabel)
            totalCount = label_pTC + 1
        weighted_num = np.sum(np.multiply(output_pTC_wpt,weights_pTC))
        weighted_den = np.sum(np.multiply(totalCount,weights_pTC))
        eff.append(weighted_num/weighted_den)
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.plot(binX,eff)
    plt.grid()
    plt.ylim(0,1)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('{} (GeV)'.format(varLabel))
    plt.savefig(plotname, dpi=fig.dpi)

def plotByBin(binVar,binVarBins,histVar,xlabel,varLab,outDir,plotName,xlim,weights=None):
    binWidth = binVarBins[1] - binVarBins[0]
    fig, ax = plt.subplots(figsize=(12, 8))
    for j in range(len(binVarBins)):
        binVal = binVarBins[j]
        if j < len(binVarBins)-1:
            cond = np.absolute(binVal-binVar) < binWidth/2.
            lab = '{:.2f} < {} < {:.2f}'.format(binVarBins[j]-binWidth/2.,varLab,binVarBins[j]+binWidth/2)
        else:
            cond = binVar > binVal - binWidth/2.
            lab = '{} > {:.2f}'.format(varLab,binVarBins[j]-binWidth/2.)
        wVL = weights[cond]
        histVL = histVar[cond]
        if len(histVL) > 0:
            histMakePlot(histVL,bins=50,weights=wVL,color=mplColors[j%len(mplColors)],facecolorOn=False,alpha=0.5,label=lab)
    ax.set_ylabel('Norm Events')
    ax.set_xlabel(xlabel)
    plt.legend()
    plt.xlim(xlim[0],xlim[1])
    plt.savefig(outDir + "/{}.pdf".format(plotName), dpi=fig.dpi)
    plt.yscale("log")
    plt.savefig(outDir + "/{}_log.pdf".format(plotName), dpi=fig.dpi)

def main():
    # parse arguments
    parser = ArgumentParser(config_options=MagiConfigOptions(strict = True, default="logs/config_out.py"),formatter_class=ArgumentDefaultsRawHelpFormatter)
    parser.add_argument("--outf", type=str, default="logs", help='Name of folder to be used to store outputs')
    parser.add_argument("--model", type=str, default="net.pth", help="Existing model to continue training, if applicable")
    parser.add_argument("--pIn", action="store_true", help="Plot input variables and their correlation.")
    parser.add_config_only(*c.config_schema)
    parser.add_config_only(**c.config_defaults)
    args = parser.parse_args()
    modelLocation = "{}/{}".format(args.outf,args.model)

    # Choose cpu or gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    if device.type == 'cuda':
        gpuIndex = torch.cuda.current_device()
        print("Using GPU named: \"{}\"".format(torch.cuda.get_device_name(gpuIndex)))

    # Load dataset
    print('Loading dataset...')
    dSet = args.dataset
    sigFiles = dSet.signal
    inputFiles = dSet.background
    hyper = args.hyper
    inputFiles.update(sigFiles)
    varSet = args.features.train
    pTBins = hyper.pTBins
    uniform = args.features.uniform
    mT = args.features.mT
    weight = args.features.weight
    dataset = RootDataset(inputFolder=dSet.path, root_file=inputFiles, variables=varSet, pTBins=pTBins, uniform=uniform, mT=mT, weight=weight)
    sizes = get_sizes(len(dataset), dSet.sample_fractions)
    train, val, test = udata.random_split(dataset, sizes, generator=torch.Generator().manual_seed(42))
    # Build model
    model = DNN_GRF(n_var=len(varSet),  n_layers_features=hyper.num_of_layers_features, n_layers_tag=hyper.num_of_layers_tag, n_layers_pT=hyper.num_of_layers_pT, n_nodes=hyper.num_of_nodes, n_outputs=2, n_pTBins=hyper.n_pTBins, drop_out_p=hyper.dropout).to(device=device)
    print("Loading model from file " + modelLocation)
    model.load_state_dict(torch.load(modelLocation))
    model.eval()
    model.to('cpu')
    label_train, input_train, output_train_tag, output_train_pTClass, pTLab_train, pT_train, mT_train, w_train = getNNOutput(train, model)
    label_test, input_test, output_test_tag, output_test_pTClass, pTLab_test, pT_test, mT_test, w_test = getNNOutput(test, model)
    fpr_Train, tpr_Train, auc_Train = getROCStuff(label_train, output_train_tag, w_train)
    fpr_Test, tpr_Test, auc_Test = getROCStuff(label_test, output_test_tag, w_test)
    bkgOnly_train = label_train == 0
    sigOnly_train = np.logical_not(bkgOnly_train)
    bkgOnly_test = label_test == 0
    sigOnly_test = np.logical_not(bkgOnly_test)

    np.savez("pTTest",pT_train=pT_train,pTLab_train=pTLab_train)

    # Creating a pandas dataFrame for training data
    df = pd.DataFrame(data=input_train,columns=varSet)
    # # testing pT prediction with GR turned off
    print(output_train_pTClass)
    predictedpT = np.argmax(output_train_pTClass,axis=1)
    truepT = pTLab_train
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, hspace=0,height_ratios=[5,1])
    axs = gs.subplots(sharex=True)
    binwidth = 1
    bins = np.arange(0,len(pTBins),binwidth)-0.5*binwidth
    truepT_bins,truepT_hist = histMake(truepT,bins,weights=w_train,norm=False)
    prepT_bins,prepT_hist = histMake(predictedpT,bins,weights=w_train,norm=False)
    np.savez("ptPredTrue",truepT_hist=truepT_hist,prepT_hist=prepT_hist,pTBins=pTBins)
    pTBP = []
    for i in range(len(pTBins)):
        if i == len(pTBins)-1:
            pTBP.append(pTBins[i])
        else:
            pTBP.append(np.mean([pTBins[i],pTBins[i+1]]))
    histplot(truepT_hist,pTBins,"xkcd:blue","True pT",alpha=1.0,hatch="//",facecolorOn=False,ax=axs[0])
    histplot(prepT_hist,pTBP,"xkcd:red","Predicted pT",points=True,ax=axs[0])
    axs[1].plot(pTBP[:-1],np.divide(prepT_hist,truepT_hist)[:-1],marker=".")
    axs[1].set_ylim(0,2)
    axs[1].set_yticks(np.arange(0,2,0.5))
    axs[0].set_yscale("log")
    axs[0].grid()
    axs[1].grid()
    axs[0].legend()
    axs[0].set_ylabel('Norm Events')
    plt.xlabel('pT (GeV)')
    # plt.xticks(ticks=bins[:-1]+binwidth*0.5,labels=[str(p) for p in pTBins[:-1]])
    # Hide x labels and tick labels for all but bottom plot.
    for ax in axs:
        ax.label_outer()
    plt.savefig(args.outf + "/predictedpTvstruepT.pdf", dpi=fig.dpi)
    print("Finished making pT prediction plot")
    # raise ValueError('Trying to stop the code here.')

    if args.pIn:
        # plot correlation
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
        df["label"] = label_train
        df["weights"] = w_train
        for var in varSet:
            dataSig = df[var][sigOnly_train]
            dataBkg = df[var][bkgOnly_train]
            fig, ax = plt.subplots(figsize=(12, 8))
            histMakePlot(dataSig,bins=50,color='xkcd:blue',alpha=0.5,label='Background')
            histMakePlot(dataBkg,bins=50,color='xkcd:red',alpha=1.0,label='Signal',hatch="//",facecolorOn=False)
            ax.set_ylabel('Norm Events')
            ax.set_xlabel(var)
            plt.legend()
            plt.savefig(args.outf + "/{}.pdf".format(var), dpi=fig.dpi)

    # plot ROC curve
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

    # plot eff vs pT
    plotEffvsVar(np.arange(250,3000,500),pT_train[bkgOnly_train],w_train[bkgOnly_train],label_train[bkgOnly_train],output_train_tag[bkgOnly_train],"pT",args.outf,wpt=0.5,sig=False)
    plotEffvsVar(np.arange(250,3000,500),pT_train[sigOnly_train],w_train[sigOnly_train],label_train[sigOnly_train],output_train_tag[sigOnly_train],"pT",args.outf,wpt=0.5)
    plotEffvsVar(np.arange(250,3000,100),mT_train[bkgOnly_train],w_train[bkgOnly_train],label_train[bkgOnly_train],output_train_tag[bkgOnly_train],"mT",args.outf,wpt=0.5,sig=False)
    plotEffvsVar(np.arange(250,3000,100),mT_train[sigOnly_train],w_train[sigOnly_train],label_train[sigOnly_train],output_train_tag[sigOnly_train],"mT",args.outf,wpt=0.5)

    # histogram NN score
    fig, ax = plt.subplots(figsize=(12, 8))
    histMakePlot(output_train_tag,bins=50,weights=w_train,color='xkcd:blue',alpha=0.5,label='Training set')
    ax.set_ylabel('Norm Events')
    ax.set_xlabel("NN Score")
    plt.legend()
    plt.savefig(args.outf + "/SNN.pdf", dpi=fig.dpi)

    # NN score per pT bin
    plotByBin(binVar=pT_train,binVarBins = np.arange(250,3000,500),histVar=output_train_tag,xlabel="NN Score",varLab="pT",outDir=args.outf,plotName="SNNperpT",xlim=[0,1],weights=w_train)
    # pT per NN score bin
    plotByBin(binVar=output_train_tag,binVarBins = np.arange(0.1,1,0.2),histVar=pT_train,xlabel="Jet $p_T (GeV)$",varLab="SNN",outDir=args.outf,plotName="pTperSNN",xlim=[0,2000],weights=w_train)
    plotByBin(binVar=output_train_tag[bkgOnly_train],binVarBins = np.arange(0.1,1,0.2),histVar=pT_train[bkgOnly_train],xlabel="Jet $p_T (GeV)$",varLab="SNN",outDir=args.outf,plotName="pTperSNN_bkg",xlim=[0,2000],weights=w_train[bkgOnly_train])
    plotByBin(binVar=output_train_tag[sigOnly_train],binVarBins = np.arange(0.1,1,0.2),histVar=pT_train[sigOnly_train],xlabel="Jet $p_T (GeV)$",varLab="SNN",outDir=args.outf,plotName="pTperSNN_sig",xlim=[0,2000],weights=w_train[sigOnly_train])
    # NN score per mT bin
    plotByBin(binVar=mT_train,binVarBins = np.arange(125,3000,250),histVar=output_train_tag,xlabel="NN Score",varLab="mT",outDir=args.outf,plotName="SNNpermT",xlim=[0,1],weights=w_train)
    # mT per NN score bin
    plotByBin(binVar=output_train_tag,binVarBins = np.arange(0.1,1,0.2),histVar=mT_train,xlabel="Jet $m_T (GeV)$",varLab="SNN",outDir=args.outf,plotName="mTperSNN",xlim=[1500,4000],weights=w_train)
    plotByBin(binVar=output_train_tag[bkgOnly_train],binVarBins = np.arange(0.1,1,0.2),histVar=mT_train[bkgOnly_train],xlabel="Jet $m_T (GeV)$",varLab="SNN",outDir=args.outf,plotName="mTperSNN_bkg",xlim=[1500,4000],weights=w_train[bkgOnly_train])
    plotByBin(binVar=output_train_tag[sigOnly_train],binVarBins = np.arange(0.1,1,0.2),histVar=mT_train[sigOnly_train],xlabel="Jet $m_T (GeV)$",varLab="SNN",outDir=args.outf,plotName="mTperSNN_sig",xlim=[1500,4000],weights=w_train[sigOnly_train])

    # 2D histogram NN vs pT
    plt.figure(figsize=(12, 8))
    axes = plt.gca()
    corr = np.corrcoef(pT_train,output_train_tag)[0][1]
    plt.text(0.7,1.05,"Correlation = {:.3f}".format(corr),transform = axes.transAxes)
    plt.hist2d(pT_train,output_train_tag,bins=[np.linspace(150,1250,40),np.linspace(0,0.15,40)])
    plt.xlabel("Jet pT (GeV)")
    plt.ylabel("NN Score")
    plt.colorbar()
    plt.savefig(args.outf + "/2D_NNvspT.pdf", dpi=fig.dpi)

    # plot discriminator
    bins = np.linspace(0, 1, 100)
    fig, ax = plt.subplots(figsize=(6, 6))
    y_Train_Sg, y_Train_Bg, w_Train_Sg, w_Train_Bg = getSgBgOutputs(label_train, output_train_tag,w_train)
    y_test_Sg, y_test_Bg, w_test_Sg, w_test_Bg = getSgBgOutputs(label_test, output_test_tag,w_test)
    ax.set_title('')
    ax.set_ylabel('Norm Events')
    ax.set_xlabel('Discriminator')
    histMakePlot(y_test_Sg,bins,weights=w_test_Sg,color='xkcd:red',alpha=1.0,label='Sg Test',hatch="//",facecolorOn=False)
    histMakePlot(y_test_Bg,bins,weights=w_test_Bg,color='xkcd:blue',alpha=0.5,label='Bg Test')
    histMakePlot(y_Train_Sg,bins,weights=w_Train_Sg,color='xkcd:red',label='Sg Train',points=True)
    histMakePlot(y_Train_Bg,bins,weights=w_Train_Bg,color='xkcd:blue',label='Bg Train',points=True)
    ax.legend(loc='best', frameon=False)
    fig.savefig(args.outf + "/discriminator.pdf", dpi=fig.dpi)

if __name__ == "__main__":
    main()
