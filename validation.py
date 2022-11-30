import torch
import torch.nn as nn
from torch.nn import functional as f
import torch.utils.data as udata
from torch.cuda.amp import autocast
import os
import particlenet_pf
from dataset import RootDataset, get_sizes
import matplotlib as mpl
import matplotlib.pyplot as plt
from magiconfig import ArgumentParser, MagiConfigOptions, ArgumentDefaultsRawHelpFormatter
from configs import configs as c
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, roc_auc_score
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm
import itertools
import copy
from GPUtil import showUtilization as gpu_usage

mpl.rc("font", family="serif", size=18)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
mplColors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

def kstest(data_train,data_test):
    ks, pv = stats.ks_2samp(data_train,data_test)
    return pv

def ztest(X1,X2):
    return abs(np.mean(X1)-np.mean(X2))/np.sqrt(np.var(X1,ddof=1) + np.var(X2,ddof=1))


def collectiveKS(dataList):
    allComs = list(itertools.combinations(range(len(dataList)),2))
    print("len(dataList)",len(dataList))
    print("len(allComs)",len(allComs))
    ksSum = 0
    for com in allComs:
        data1 = np.array(dataList[com[0]])
        data2 = np.array(dataList[com[1]])
        newdata1 = data1[(data1>0) & (data2>0)]
        newdata2 = data2[(data1>0) & (data2>0)]
        ks, pv = stats.ks_2samp(newdata1,newdata2)
        ksSum += pv
    return ksSum/len(allComs)

def getCondition(inputFileNames,key,inpIndex):
    conditions = np.zeros(len(inpIndex),dtype=bool)
    for i in range(len(inputFileNames)):
        fileName = inputFileNames[i]
        if key in fileName:
            conditions = conditions | (inpIndex == i)
    return conditions

def getNNOutput(dataset, model, device, signalIndex=2):
    batchSize = 100
    labels = np.array([])
    binaryLabels = np.array([])
    output_tags = np.array([])
    binaryOutputs = np.array([])
    inputFileIndices = np.array([])
    pT = np.array([])
    weight = np.array([])
    meds = np.array([])
    darks = np.array([])
    rinvs = np.array([])
    alphas = np.array([])
    loader = udata.DataLoader(dataset=dataset, batch_size=batchSize, num_workers=0)
    for i, data in tqdm(enumerate(loader), unit="batch", total=len(loader)):
        l, points, features, jetFeatures, inputFileIndex, p, w, med, dark, rinv, alpha = data
        labels = np.concatenate((labels,l.squeeze(1).numpy()))
        inputFileIndices = np.concatenate((inputFileIndices,inputFileIndex.squeeze(1).numpy()))
        pT = np.concatenate((pT,p.squeeze(1).float().numpy()))
        weight = np.concatenate((weight,w.squeeze(1).float().numpy()))
        meds = np.concatenate((meds,med.squeeze(1).float().numpy()))
        darks = np.concatenate((darks,dark.squeeze(1).float().numpy()))
        rinvs = np.concatenate((rinvs,rinv.squeeze(1).float().numpy()))
        alphas = np.concatenate((alphas,alpha.squeeze(1).numpy()))
        model.eval()
        inputPoints = points.float()
        inputFeatures = features.float()
        inputJetFeatures = jetFeatures.float()
        with autocast():
            out_tag = model(inputPoints.to(device),inputFeatures.to(device))
            outSoftmax = f.softmax(out_tag,dim=1)

            binaryOutput = outSoftmax[:,signalIndex].cpu().detach().numpy()
            binaryOutputs = np.concatenate((binaryOutputs,binaryOutput))
            binaryLabel = (l.squeeze(1).numpy()==signalIndex).astype(int)
            binaryLabels = np.concatenate((binaryLabels,binaryLabel))

            output_tag = outSoftmax[:,signalIndex].cpu().detach().numpy()
            output_tags = np.concatenate((output_tags,output_tag))
    return binaryLabels, binaryOutputs, inputFileIndices, pT, weight, meds, darks, rinvs, alphas

def getROCStuff(label, output, weights=None):
    fpr, tpr, thresholds = roc_curve(label, output, sample_weight=weights)
    auc = roc_auc_score(label, output, sample_weight=weights)
    return fpr, tpr, auc

def getSgBgOutputs(sigCond, bkgCond, output, weights):
    y_Sg = output[sigCond]
    y_Bg = output[bkgCond]
    w_Sg = weights[sigCond]
    w_Bg = weights[bkgCond]

    return y_Sg, y_Bg, w_Sg, w_Bg

def histMake(data,binEdge,weights=None,norm=True):
    data,bins = np.histogram(data, bins=binEdge, weights=weights, density=norm)
    bins = np.array([0.5 * (bins[i] + bins[i+1]) for i in range(len(bins)-1)])
    binwidth = bins[1] - bins[0]
    pbins = np.append(bins,bins[-1]+binwidth)
    pdata = np.append(data,data[-1])
    return np.array(pbins),np.array(pdata)

def histplot(pdata,pbins,color,label,alpha=1.0,hatch=None,points=False,facecolorOn=True,ax=plt):
    if points:
        ax.plot(pbins[:-1]+(pbins[1]-pbins[0])/2.,pdata[:-1],color=color,label=label,marker=".",linestyle="None")
    else:
        ax.step(pbins,pdata,where="post",color=color)
        if facecolorOn:
            facecolor=color
        else:
            facecolor="none"
        ax.fill_between(pbins,pdata, step="post", edgecolor=color, facecolor=facecolor, label=label, alpha=alpha, hatch=hatch)

def histMakePlot(data,binEdge,color,label,weights=None,alpha=1.0,hatch=None,points=False,facecolorOn=True,norm=True,ax=plt):
    pbins,pdata = histMake(data,binEdge,weights=weights,norm=norm)
    histplot(pdata,pbins,color,label,alpha=alpha,hatch=hatch,points=points,facecolorOn=facecolorOn,ax=ax)
    return pdata,pbins

# signal vs. background figure of merit
def fom(S,B):
    return np.sqrt(2 * ( (S+B) * np.log(1+S/B) - S) )

def plotSignificance(cutBins,plotBinEdge,var_train,w_train,inpIndex_train,output_train_tag,tagLabel,varLabel,outf,wpt=0.5):
    foms = []
    binWidth = cutBins[1]-cutBins[0]
    bkgTag = np.logical_and(output_train_tag > wpt,inpIndex_train >= 2)
    sigTag = np.logical_and(output_train_tag > wpt,inpIndex_train <= 1) # inpIndex_train == 1 means using only baseline signal, inpIndex_train <= 1 means all signals
    for j in range(len(cutBins)):
        binVal = cutBins[j]
        if j < len(cutBins)-1:
            varCond = np.absolute(binVal-var_train) < binWidth/2.
        else:
            varCond = var_train > binVal - binWidth/2.
        bkg_var = np.multiply(np.where(bkgTag,1,0),w_train)[varCond]
        sig_var = np.multiply(np.where(sigTag,1,0),w_train)[varCond]
        weighted_sig = np.sum(sig_var)
        weighted_bkg = np.sum(bkg_var)
        foms.append(fom(weighted_sig,weighted_bkg))
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.step(cutBins,foms,where="mid")
    axes = plt.gca()
    plt.text(0.1,0.8,"Max Sig. = {:.2f}".format(np.amax(foms)),transform = axes.transAxes)
    plt.grid()
    # plt.ylim(0,np.nanmax(eff)*1.1)
    plt.ylabel("FOM ( sqrt(2((S+B)*log(1+S/B)-S)) )")
    plt.xlabel('{} (GeV)'.format(varLabel))
    plt.savefig(outf + "/significance_{}.png".format(varLabel), dpi=fig.dpi, bbox_inches="tight")
    plt.figure(figsize=(12, 8))
    bkgOnly_train = tagLabel == 0
    sigOnly_train = np.logical_not(bkgOnly_train)
    histMakePlot(var_train[sigOnly_train],plotBinEdge,weights=w_train[sigOnly_train],color='xkcd:red',alpha=1.0,label='Sg Train',hatch="//",facecolorOn=False)
    histMakePlot(var_train[bkgOnly_train],plotBinEdge,weights=w_train[bkgOnly_train],color='xkcd:blue',alpha=0.5,label='Bkg Train',hatch="//")
    plt.ylabel("Event")
    plt.xlabel('{} (GeV)'.format(varLabel))
    plt.legend()
    plt.savefig(outf + "/sigVsBkg_{}.png".format(varLabel), dpi=fig.dpi, bbox_inches="tight")

def plotEffvsVar(binX,var_train,w_train,label_train,output_train_tag,varLabel,outf,dfOut=None,sig=True):
    fig, ax = plt.subplots(figsize=(12, 8))
    binWidth = binX[1]-binX[0]
    for wpt in [0.05,0.1,0.2,0.3,0.5]:
        eff = []
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
                plotname = outf + "/sigEffVs{}.png".format(varLabel)
                totalCount = label_pTC
            else:
                ylabel = "Mistag Rate"
                plotname = outf + "/mistagVs{}.png".format(varLabel)
                totalCount = label_pTC + 1
            weighted_num = np.sum(np.multiply(output_pTC_wpt,weights_pTC))
            weighted_den = np.sum(np.multiply(totalCount,weights_pTC))
            if weighted_den > 0:
                eff.append(weighted_num/weighted_den)
            else:
                eff.append(0)
        binX_ori = np.array(binX)
        eff_ori = np.array(eff)
        binX = binX_ori[eff_ori>0]
        eff = eff_ori[eff_ori>0]
        plt.plot(binX,eff,label="wpt={}".format(wpt))
        if type(dfOut) == pd.core.frame.DataFrame:
            corr = np.corrcoef(binX,eff)
            dfOut["bkgflatness_{}_wpt{}".format(varLabel,wpt)] = [abs(corr[0][1])]
    plt.grid()
    if sig:
        ylabel = "Signal Efficiency"
        plotname = outf + "/sigEffVs{}.png".format(varLabel)
        plt.ylim(0,1.0)
    else:
        ylabel = "Mistag Rate"
        plotname = outf + "/mistagVs{}.png".format(varLabel)
        plt.ylim(0,0.4)
    # plt.ylim(0,np.nanmax(eff)*1.1)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('{} (GeV)'.format(varLabel))
    plt.legend()
    plt.savefig(plotname, dpi=fig.dpi, bbox_inches="tight")

def plotByBin(binVar,binVarBins,histVar,xlabel,varLab,outDir,plotName,xlim,disLab=False,weights=None,histBinEdge=50,dfOut=None):
    binWidth = binVarBins[1] - binVarBins[0]
    for weightCondition, weights in [["weighted", weights], ["unweighted", np.ones(len(weights))]]:
        fig, ax = plt.subplots(figsize=(12, 8))
        hDataList = []
        for j in range(len(binVarBins)):
            binVal = binVarBins[j]
            if j < len(binVarBins)-1:
                cond = np.absolute(binVal-binVar) < binWidth/2.
                if disLab:
                    lab = '{} = {:.2f}'.format(varLab,binVarBins[j])
                else:
                    lab = '{:.2f} < {} < {:.2f}'.format(binVarBins[j]-binWidth/2.,varLab,binVarBins[j]+binWidth/2)
            else:
                cond = binVar > binVal - binWidth/2.
                if disLab:
                    lab = '{} = {:.2f}'.format(varLab,binVarBins[j])
                else:
                    lab = '{:.2f} < {} < {:.2f}'.format(binVarBins[j]-binWidth/2.,varLab,binVarBins[j]+binWidth/2)
            wVL = weights[cond]
            histVL = histVar[cond]
            if plotName=="SNNpermdark_sig":
                print(plotName)
                print(cond)
                print(np.unique(histVL))
            if len(histVL) > 0:
                histData,hBins = histMakePlot(histVL,binEdge=histBinEdge,weights=wVL,color=mplColors[j%len(mplColors)],facecolorOn=False,alpha=0.5,label=lab)
                if plotName=="SNNpermdark_sig":
                    print(histData)
                    print(hBins)
                if type(dfOut) == pd.core.frame.DataFrame:
                    hDataList.append(histData)
        if len(hDataList) > 1:
            dfOut["avgKS_{}_{}".format(plotName,weightCondition)] = collectiveKS(hDataList)
        ax.set_ylabel('Norm Events')
        ax.set_xlabel(xlabel)
        plt.legend()
        plt.xlim(xlim[0],xlim[1])
        plt.savefig(outDir + "/{}_{}.png".format(plotName,weightCondition), dpi=fig.dpi, bbox_inches="tight")
        plt.yscale("log")
        # plt.ylim(0.0001,0.2)
        plt.savefig(outDir + "/{}_{}_log.png".format(plotName,weightCondition), dpi=fig.dpi, bbox_inches="tight")

def NNvsVar2D(var_train,output_train_tag,var_range,tag_range,xlabel,plotType,outDir,dfOut=None):
    plt.figure(figsize=(12, 8))
    axes = plt.gca()
    corr = np.corrcoef(var_train,output_train_tag)[0][1]
    plt.text(0.7,1.05,"Correlation = {:.3f}".format(corr),transform = axes.transAxes)
    varLabel = ""
    if "p_T" in xlabel:
        varLabel = "pT"
    if type(dfOut) == pd.core.frame.DataFrame:
        dfOut["pearsonCorr_{}_{}".format(varLabel,plotType)] = abs(corr)
    plt.hist2d(var_train,output_train_tag,bins=[var_range,tag_range],norm=mpl.colors.LogNorm())
    plt.xlabel(xlabel)
    plt.ylabel("NN Score")
    plt.colorbar()
    plt.savefig(outDir + "/2D_NNvs{}_{}.png".format(varLabel,plotType), bbox_inches="tight")

def plotDiscriminator(sigCond_train, bkgCond_train, sigCond_test, bkgCond_test, output_train_tag, w_train, output_test_tag, w_test, outputFolder, dfOut, colorSg, colorBg, bkgLab="Bg"):
    binEdge = np.arange(-0.01, 1.02, 0.02)
    for weightCondition, w_train, w_test in [["weighted", w_train, w_test], ["unweighted", np.ones(len(w_train)), np.ones(len(w_test))]]:
        fig, ax = plt.subplots(figsize=(6, 6))
        y_Train_Sg, y_Train_Bg, w_Train_Sg, w_Train_Bg = getSgBgOutputs(sigCond_train, bkgCond_train, output_train_tag,w_train)
        y_test_Sg, y_test_Bg, w_test_Sg, w_test_Bg = getSgBgOutputs(sigCond_test, bkgCond_test, output_test_tag,w_test)
        ax.set_title('')
        ax.set_ylabel('Norm Events')
        ax.set_xlabel('Discriminator')
        ax.set_xlim(0,1.05)
        h_test_Sg,h_test_Sg_Bins = histMakePlot(y_test_Sg,binEdge,weights=w_test_Sg,color=colorSg,alpha=1.0,label='Sg Test',hatch="//",facecolorOn=False)
        h_test_Bg,h_test_Bg_Bins = histMakePlot(y_test_Bg,binEdge,weights=w_test_Bg,color=colorBg,alpha=0.5,label='{} Test'.format(bkgLab))
        h_Train_Sg,h_Train_Sg_Bins = histMakePlot(y_Train_Sg,binEdge,weights=w_Train_Sg,color=colorSg,label='Sg Train',points=True)
        h_Train_Bg,h_Train_Bg_Bins = histMakePlot(y_Train_Bg,binEdge,weights=w_Train_Bg,color=colorBg,label='{} Train'.format(bkgLab),points=True)
        dfOut["sigovertrain_{}".format(weightCondition)] = [kstest(h_Train_Sg,h_test_Sg)]
        dfOut["{}overtrain_{}".format(bkgLab,weightCondition)] = [kstest(h_Train_Bg,h_test_Bg)]
        ax.legend(loc='best', frameon=False)
        fig.savefig("{}/discriminator_{}_{}.png".format(outputFolder,bkgLab,weightCondition), dpi=fig.dpi, bbox_inches="tight")
        # ks test for overtraining

def plotROC(label_train, output_train_tag, w_train, label_test, output_test_tag, w_test, dfOut, outFolder, bkgLab):
    for weightCondition, w_train, w_test in [["weighted", w_train, w_test], ["unweighted", np.ones(len(w_train)), np.ones(len(w_test))]]:
        fpr_Train, tpr_Train, auc_Train = getROCStuff(label_train, output_train_tag, w_train)
        fpr_Test, tpr_Test, auc_Test = getROCStuff(label_test, output_test_tag, w_test)
        fig = plt.figure()
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve', pad=45.0)
        plt.plot(fpr_Train, tpr_Train, label="Train (area = {:.3f})".format(auc_Train), color='xkcd:red')
        plt.plot(fpr_Test, tpr_Test, label="Test (area = {:.3f})".format(auc_Test), color='xkcd:black')
        plt.legend(loc='best')
        dfOut["auc_train_{}_{}".format(bkgLab,weightCondition)] = [auc_Train]
        dfOut["auc_test_{}_{}".format(bkgLab,weightCondition)] = [auc_Test]
        fig.savefig("{}/roc_plot_{}_{}.png".format(outFolder,bkgLab,weightCondition), dpi=fig.dpi, bbox_inches="tight")
        plt.close(fig)

def main():
    # parse arguments
    parser = ArgumentParser(config_options=MagiConfigOptions(strict = True, default="logs/config_out.py"),formatter_class=ArgumentDefaultsRawHelpFormatter)
    parser.add_argument("--outf", type=str, default="logs", help='Name of folder to be used to store outputs')
    parser.add_argument("--inpf", type=str, default="processedData_nc100", help='Name of the npz input training file')
    parser.add_argument("--model", type=str, default="net.pth", help="Existing model to continue training, if applicable")
    parser.add_argument("--pIn", action="store_true", help="Plot input variables and their correlation.")
    parser.add_config_only(*c.config_schema)
    parser.add_config_only(**c.config_defaults)
    args = parser.parse_args()
    modelLocation = "{}/{}".format(args.outf,args.model)
    print("Model location:",modelLocation)

    if not os.path.isdir(args.outf):
        os.mkdir(args.outf)
    # Choose cpu or gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    if device.type == 'cuda':
        gpuIndex = torch.cuda.current_device()
        print("Using GPU named: \"{}\"".format(torch.cuda.get_device_name(gpuIndex)))

    colorDict = {
        "allBkg":'xkcd:blue',
        "allSig":'xkcd:red',
        "ZJets": "#999999",
        "WJets": "#cccc19",
        "TTJets": "#6666cc",
        "QCD": "#4bd42d",
        "mMed600": "#cd2bcc",
        "baseline": "#cc660d",
        "M-2000_mD-1": "#f2231b",
        "M-2000_mD-100": "#9a27cc",
        "M-2000_r-0p1": "#663303",
        "M-2000_r-0p7": "#f69acc",
        "M-2000_a-low": "#cccccc",
        "M-2000_a-high": "#999910",
        "mMed3000": "#5efdff",
    }

    dfOut = pd.DataFrame()
    # Load dataset
    print('Loading dataset...')
    dSet = args.dataset
    sigFiles = dSet.signal
    inputFiles = dSet.background
    hyper = args.hyper
    inputFiles.update(sigFiles)
    varSetjetConst = args.features.jetConst
    varSetjetVariables = args.features.jetVariables
    varSetjetVariables = []
    for var in args.features.jetVariables:
        if var not in ['jCstPtAK8', 'jCstEtaAK8', 'jCstPhiAK8', 'jCstEnergyAK8']:
            varSetjetVariables.append(var)
    print("Input jet features:",varSetjetVariables)
    pTBins = hyper.pTBins
    uniform = args.features.uniform
    weight = args.features.weight
    numConst = args.hyper.numConst
    train = RootDataset("processedDataNPZ/processedData_nc100_train.npz")
    test = RootDataset("processedDataNPZ/processedData_nc100_test.npz")
    inputFeatureVars = train.inputFeaturesVarName
    print("Input jet constituent features:",inputFeatureVars)
    inputFileNames = train.inputFileNames
    # Build model
    # Build model
    network_module = particlenet_pf
    network_options = {}
    network_options["num_of_k_nearest"] = args.hyper.num_of_k_nearest
    network_options["num_of_edgeConv_dim"] = args.hyper.num_of_edgeConv_dim
    network_options["num_of_edgeConv_convLayers"] = args.hyper.num_of_edgeConv_convLayers
    network_options["num_of_fc_layers"] = args.hyper.num_of_fc_layers
    network_options["num_of_fc_nodes"] = args.hyper.num_of_fc_nodes
    network_options["fc_dropout"] = args.hyper.fc_dropout
    network_options["num_classes"] = args.hyper.num_classes
    model = network_module.get_model(inputFeatureVars,**network_options)
    model = copy.deepcopy(model)
    print("Loading model from file " + modelLocation)
    model.load_state_dict(torch.load(modelLocation))
    model.eval()
    model.to(device)
    label_train, output_train_tag, inpIndex_train, pT_train, w_train, med_train, dark_train, rinv_train, alpha_train = getNNOutput(train, model, device)
    label_test, output_test_tag, inpIndex_test, pT_test, w_test, med_test, dark_test, rinv_test, alpha_test = getNNOutput(test, model, device)
    print("pT_train max:",np.amax(pT_train))
    print("pT_test max:",np.amax(pT_test))
    QCD_train = getCondition(inputFileNames,"QCD",inpIndex_train)
    TTJets_train = getCondition(inputFileNames,"TTJets",inpIndex_train)
    signal_train = getCondition(inputFileNames,"SVJ",inpIndex_train)
    QCD_test = getCondition(inputFileNames,"QCD",inpIndex_test)
    TTJets_test = getCondition(inputFileNames,"TTJets",inpIndex_test)
    signal_test = getCondition(inputFileNames,"SVJ",inpIndex_test)
    bkgOnly_train = label_train == 0
    sigOnly_train = np.logical_not(bkgOnly_train)
    bkgOnly_test = label_test == 0
    sigOnly_test = np.logical_not(bkgOnly_test)
    wpt = 0.4
    bkgtrain_NNOut = output_train_tag < wpt
    sigtrain_NNOut = np.logical_not(bkgtrain_NNOut)

    # Creating a pandas dataFrame for training data
    # df = pd.DataFrame(data=inputFeatures_train,columns=varSet)
    # # testing pT prediction with GR turned off

    # if args.pIn:
    #     # plot correlation
    #     fig, ax = plt.subplots(figsize=(12, 8))
    #     corr = np.round(df.corr(),2)
    #     ax = sns.heatmap(corr,cmap="Spectral")
    #     bottom, top = ax.get_ylim()
    #     left, right = ax.get_xlim()
    #     ax.set_ylim(bottom + 0.5, top - 0.5)
    #     ax.set_xlim(left - 0.5, right + 0.5)
    #     plt.savefig(args.outf + "/corrHeatMap.png", dpi=fig.dpi, bbox_inches="tight")
    #     fig, ax = plt.subplots(figsize=(12, 8))
    #
    #     # plot input variable
    #     zTests = pd.DataFrame(columns=["var","ztest"])
    #     df["label"] = label_train
    #     df["weights"] = w_train
    #     for var in varSet:
    #         dataSig = df[var][sigOnly_train]
    #         dataBkg = df[var][bkgOnly_train]
    #         plt.figure(figsize=(12, 8))
    #         dataAll = dataSig,dataBkg),axis=None)
    #         minX = np.amin(dataAll)
    #         maxX = np.amax(dataAll)
    #         binX = np.linspace(minX,maxX,50)
    #         histMakePlot(dataSig,binEdge=binX,color='xkcd:blue',alpha=0.5,label='Background')
    #         histMakePlot(dataBkg,binEdge=binX,color='xkcd:red',alpha=1.0,label='Signal',hatch="//",facecolorOn=False)
    #         plt.ylabel('Norm Events')
    #         plt.xlabel(var)
    #         plt.legend()
    #         plt.savefig(args.outf + "/{}.png".format(var), dpi=fig.dpi, bbox_inches="tight")
    #         zt = ztest(dataSig,dataBkg)
    #         zTests.loc[len(zTests.index)] = [var,zt]
    #         zTests.to_csv("{}/zTests.csv".format(args.outf))

    QCD_train = getCondition(inputFileNames,"QCD",inpIndex_train)
    TTJets_train = getCondition(inputFileNames,"TTJets",inpIndex_train)
    signal_train = getCondition(inputFileNames,"SVJ",inpIndex_train)
    QCD_test = getCondition(inputFileNames,"QCD",inpIndex_test)
    TTJets_test = getCondition(inputFileNames,"TTJets",inpIndex_test)
    signal_test = getCondition(inputFileNames,"SVJ",inpIndex_test)
    QCD_Sig_train = signal_train | QCD_train
    QCD_Sig_test = signal_test | QCD_test
    TTJets_Sig_train = signal_train | TTJets_train
    TTJets_Sig_test = signal_test | TTJets_test

    # plot pT
    for weightCondition, weight_train in [["weighted", w_train], ["unweighted", np.ones(len(w_train))]]:
        plt.figure(figsize=(12, 8))
        minX = np.amin(pT_train)
        maxX = 3000
        binX = np.linspace(minX,maxX,100)
        histMakePlot(pT_train[signal_train],binEdge=binX,color=colorDict["allSig"],weights=weight_train[signal_train],alpha=0.5,label='All signal')
        histMakePlot(pT_train[QCD_train],binEdge=binX,color=colorDict["QCD"],weights=weight_train[QCD_train],alpha=0.5,label='QCD')
        histMakePlot(pT_train[TTJets_train],binEdge=binX,color=colorDict["TTJets"],weights=weight_train[TTJets_train],alpha=0.5,label='TTJets')
        plt.legend()
        plt.xlabel("pT")
        plt.ylabel("Norm Events")
        plt.savefig(args.outf + "/pT_{}.png".format(weightCondition), bbox_inches="tight")
        plt.yscale("log")
        plt.savefig(args.outf + "/pT_{}_log.png".format(weightCondition), bbox_inches="tight")
        # unnormalized
        plt.figure(figsize=(12, 8))
        minX = np.amin(pT_train)
        maxX = 3000
        binX = np.linspace(minX,maxX,100)
        histMakePlot(pT_train[signal_train],binEdge=binX,color=colorDict["allSig"],norm=False,weights=weight_train[signal_train],alpha=0.5,label='All signal')
        histMakePlot(pT_train[QCD_train],binEdge=binX,color=colorDict["QCD"],norm=False,weights=weight_train[QCD_train],alpha=0.5,label='QCD')
        histMakePlot(pT_train[TTJets_train],binEdge=binX,color=colorDict["TTJets"],norm=False,weights=weight_train[TTJets_train],alpha=0.5,label='TTJets')
        plt.legend()
        plt.xlabel("pT")
        plt.ylabel("Events")
        plt.savefig(args.outf + "/pT_{}_unNormed.png".format(weightCondition), bbox_inches="tight")
        plt.yscale("log")
        plt.savefig(args.outf + "/pT_{}_log_unNormed.png".format(weightCondition), bbox_inches="tight")

    
    # plot ROC curve
    plotROC(label_train, output_train_tag, w_train, label_test, output_test_tag, w_test, dfOut, args.outf, "Bg")
    plotROC(label_train[QCD_Sig_train], output_train_tag[QCD_Sig_train], w_train[QCD_Sig_train], label_test[QCD_Sig_test], output_test_tag[QCD_Sig_test], w_test[QCD_Sig_test], dfOut, args.outf, "QCD")
    plotROC(label_train[TTJets_Sig_train], output_train_tag[TTJets_Sig_train], w_train[TTJets_Sig_train], label_test[TTJets_Sig_test], output_test_tag[TTJets_Sig_test], w_test[TTJets_Sig_test], dfOut, args.outf, "TTJets")
    
    # plot eff vs pT
    plotEffvsVar(np.arange(250,3000,100),pT_train[bkgOnly_train],w_train[bkgOnly_train],label_train[bkgOnly_train],output_train_tag[bkgOnly_train],"pT",args.outf,dfOut,sig=False)
    plotEffvsVar(np.arange(250,3000,100),pT_train[sigOnly_train],w_train[sigOnly_train],label_train[sigOnly_train],output_train_tag[sigOnly_train],"pT",args.outf)
    # plot significance
    plotSignificance(np.arange(0,1,0.1),np.arange(0,1,0.1),rinv_train,w_train,inpIndex_train,output_train_tag,label_train,"rinv",args.outf,wpt=0.5)
    # histogram NN score
    fig, ax = plt.subplots(figsize=(12, 8))
    histMakePlot(output_train_tag,binEdge=50,weights=w_train,color='xkcd:blue',alpha=0.5,label='Training set')
    ax.set_ylabel('Norm Events')
    ax.set_xlabel("NN Score")
    plt.legend()
    plt.savefig(args.outf + "/SNN.png", dpi=fig.dpi, bbox_inches="tight")

    # plot discriminator
    plotDiscriminator(signal_train, (QCD_train | TTJets_train), signal_test, (QCD_test | TTJets_test), output_train_tag, w_train, output_test_tag, w_test, args.outf, dfOut, colorDict["allSig"], colorDict["allBkg"], "Bg")
    plotDiscriminator(signal_train, QCD_train, signal_test, QCD_test, output_train_tag, w_train, output_test_tag, w_test, args.outf, dfOut, colorDict["allSig"], colorDict["QCD"], "QCD")
    plotDiscriminator(signal_train, TTJets_train, signal_test, TTJets_test, output_train_tag, w_train, output_test_tag, w_test, args.outf, dfOut, colorDict["allSig"], colorDict["TTJets"], "TTJets")
    
    # NN score per pT bin
    plotByBin(binVar=pT_test,binVarBins = np.arange(250,3000,500),histVar=output_test_tag,xlabel="NN Score",varLab="pT",outDir=args.outf,plotName="SNNperpT",xlim=[0,1],weights=w_test,histBinEdge=np.arange(-0.01, 1.02, 0.02))
    plotByBin(binVar=pT_test[bkgOnly_test],binVarBins = np.arange(250,2000,500),histVar=output_test_tag[bkgOnly_test],xlabel="NN Score",varLab="pT",outDir=args.outf,plotName="SNNperpT_bkg",xlim=[0,1.02],weights=w_test[bkgOnly_test],histBinEdge=np.arange(-0.01, 1.02, 0.02),dfOut=dfOut)
    plotByBin(binVar=pT_test[sigOnly_test],binVarBins = np.arange(250,2000,500),histVar=output_test_tag[sigOnly_test],xlabel="NN Score",varLab="pT",outDir=args.outf,plotName="SNNperpT_sig",xlim=[0,1.02],weights=w_test[sigOnly_test],histBinEdge=np.arange(-0.01, 1.02, 0.02))
    plotByBin(binVar=pT_test[QCD_test],binVarBins = np.arange(250,2000,500),histVar=output_test_tag[QCD_test],xlabel="NN Score",varLab="pT",outDir=args.outf,plotName="SNNperpT_QCD",xlim=[0,1.02],weights=w_test[QCD_test],histBinEdge=np.arange(-0.01, 1.02, 0.02),dfOut=dfOut)
    plotByBin(binVar=pT_test[TTJets_test],binVarBins = np.arange(250,2000,500),histVar=output_test_tag[TTJets_test],xlabel="NN Score",varLab="pT",outDir=args.outf,plotName="SNNperpT_TTJets",xlim=[0,1.02],weights=w_test[TTJets_test],histBinEdge=np.arange(-0.01, 1.02, 0.02),dfOut=dfOut)

    # pT per NN score bin
    plotByBin(binVar=output_test_tag,binVarBins = np.arange(0.1,1,0.2),histVar=pT_test,xlabel="Jet $p_T (GeV)$",varLab="SNN",outDir=args.outf,plotName="pTperSNN",xlim=[0,3000],weights=w_test,histBinEdge=np.arange(-30,3001,60))
    plotByBin(binVar=output_test_tag[bkgOnly_test],binVarBins = np.arange(0.1,1,0.2),histVar=pT_test[bkgOnly_test],xlabel="Jet $p_T (GeV)$",varLab="SNN",outDir=args.outf,plotName="pTperSNN_bkg",xlim=[0,3000],weights=w_test[bkgOnly_test],histBinEdge=np.arange(-30,3001,60),dfOut=dfOut)
    plotByBin(binVar=output_test_tag[sigOnly_test],binVarBins = np.arange(0.1,1,0.2),histVar=pT_test[sigOnly_test],xlabel="Jet $p_T (GeV)$",varLab="SNN",outDir=args.outf,plotName="pTperSNN_sig",xlim=[0,3000],weights=w_test[sigOnly_test],histBinEdge=np.arange(-30,3001,60))
    plotByBin(binVar=output_test_tag[QCD_test],binVarBins = np.arange(0.1,1,0.2),histVar=pT_test[QCD_test],xlabel="Jet $p_T (GeV)$",varLab="SNN",outDir=args.outf,plotName="pTperSNN_QCD",xlim=[0,3000],weights=w_test[QCD_test],histBinEdge=np.arange(-30,3001,60),dfOut=dfOut)
    plotByBin(binVar=output_test_tag[TTJets_test],binVarBins = np.arange(0.1,1,0.2),histVar=pT_test[TTJets_test],xlabel="Jet $p_T (GeV)$",varLab="SNN",outDir=args.outf,plotName="pTperSNN_TTJets",xlim=[0,3000],weights=w_test[TTJets_test],histBinEdge=np.arange(-30,3001,60))
    # # NN score per rinv bin
    # plotByBin(binVar=rinv_train,binVarBins = np.arange(0.1,1,0.1),histVar=output_train_tag,xlabel="NN Score",varLab="rinv",outDir=args.outf,plotName="SNNperrinv",xlim=[0,1.02],weights=w_train,histBinEdge=np.arange(-0.01, 1.02, 0.02),disLab=True)
    # plotByBin(binVar=rinv_train[bkgOnly_train],binVarBins = np.arange(0.1,1,0.1),histVar=output_train_tag[bkgOnly_train],xlabel="NN Score",varLab="rinv",outDir=args.outf,plotName="SNNperrinv_bkg",xlim=[0,1.02],weights=w_train[bkgOnly_train],histBinEdge=np.arange(-0.01, 1.02, 0.02),disLab=True)
    # plotByBin(binVar=rinv_train[sigOnly_train],binVarBins = np.arange(0.1,1,0.1),histVar=output_train_tag[sigOnly_train],xlabel="NN Score",varLab="rinv",outDir=args.outf,plotName="SNNperrinv_sig",xlim=[0,1.02],weights=w_train[sigOnly_train],histBinEdge=np.arange(-0.01, 1.02, 0.02),dfOut=dfOut,disLab=True)
    # # rinv per NN score bin
    # plotByBin(binVar=output_train_tag,binVarBins = np.arange(0.1,1,0.2),histVar=rinv_train,xlabel="rinv",varLab="SNN",outDir=args.outf,plotName="rinvperSNN",xlim=[0,1],weights=w_train,histBinEdge=np.arange(-0.05,1.1,0.1))
    # plotByBin(binVar=output_train_tag[bkgOnly_train],binVarBins = np.arange(0.1,1,0.2),histVar=rinv_train[bkgOnly_train],xlabel="rinv",varLab="SNN",outDir=args.outf,plotName="rinvperSNN_bkg",xlim=[0,1],weights=w_train[bkgOnly_train],histBinEdge=np.arange(-0.05,1.1,0.1))
    # plotByBin(binVar=output_train_tag[sigOnly_train],binVarBins = np.arange(0.1,1,0.2),histVar=rinv_train[sigOnly_train],xlabel="rinv",varLab="SNN",outDir=args.outf,plotName="rinvperSNN_sig",xlim=[0,1],weights=w_train[sigOnly_train],histBinEdge=np.arange(-0.05,1.1,0.1),dfOut=dfOut)
    # NN score per mdark bin
    #print("dark_train",np.unique(dark_train))
    #plotByBin(binVar=dark_train,binVarBins = np.arange(10,111,10),histVar=output_train_tag,xlabel="NN Score",varLab="mdark",outDir=args.outf,plotName="SNNpermdark",xlim=[0,1.02],weights=w_train,histBinEdge=np.arange(-0.01, 1.02, 0.02),disLab=True)
    #plotByBin(binVar=dark_train[sigOnly_train],binVarBins = np.arange(10,111,10),histVar=output_train_tag[sigOnly_train],xlabel="NN Score",varLab="mdark",outDir=args.outf,plotName="SNNpermdark_sig",xlim=[0,1.02],weights=w_train[sigOnly_train],histBinEdge=np.arange(-0.01, 1.02, 0.02),dfOut=dfOut,disLab=True)
    # mdark per NN score bin
    #plotByBin(binVar=output_train_tag,binVarBins = np.arange(0.1,1,0.2),histVar=dark_train,xlabel="mdark",varLab="SNN",outDir=args.outf,plotName="mdarkperSNN",xlim=[0,120],weights=w_train,histBinEdge=np.arange(5,111,10))
    #plotByBin(binVar=output_train_tag[sigOnly_train],binVarBins = np.arange(0.1,1,0.2),histVar=dark_train[sigOnly_train],xlabel="mdark",varLab="SNN",outDir=args.outf,plotName="mdarkperSNN_sig",xlim=[0,120],weights=w_train[sigOnly_train],histBinEdge=np.arange(5,111,10),dfOut=dfOut)

    # 2D histogram NN vs pT
    # NNvsVar2D(pT_train,output_train_tag,np.linspace(100,2000,100),np.linspace(0,1.0,100),"Jet $p_T (GeV)$","",args.outf)
    # NNvsVar2D(pT_train[bkgOnly_train],output_train_tag[bkgOnly_train],np.linspace(100,2000,50),np.linspace(0,1.0,100),"Jet $p_T (GeV)$","bkg",args.outf,dfOut)
    # NNvsVar2D(pT_train[sigOnly_train],output_train_tag[sigOnly_train],np.linspace(100,2000,50),np.linspace(0,1.0,100),"Jet $p_T (GeV)$","sig",args.outf,dfOut)

    # save important quantities for assessing performance
    dfOut.to_csv("{}/output.csv".format(args.outf),index=False)
    print(dfOut)

if __name__ == "__main__":
    main()
