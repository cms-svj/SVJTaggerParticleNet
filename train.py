#!/bin/env python
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as f
# from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast
import torch.utils.data as udata
import torch.optim as optim
import os
import particlenet_pf
from dataset import RootDataset, get_sizes, splitDataSetEvenly
import matplotlib.pyplot as plt
from magiconfig import ArgumentParser, MagiConfigOptions, ArgumentDefaultsRawHelpFormatter
from configs import configs as c
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, roc_auc_score
from tqdm import tqdm
from Disco import distance_corr
import copy
from GPUtil import showUtilization as gpu_usage

# ask Kevin how to create training root files for the NN
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["NCCL_DEBUG"] = "INFO"

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def processBatch(args, device, varSet, data, model, criterion, lambdas, epoch):
    label, points, features, mcType, pTLab, pT, mT, w, med, dark, rinv, alpha = data
    l1, l2, lgr, ldc = lambdas
    print("\n Initial GPU Usage")
    gpu_usage()
    with autocast():
        # inputPoints = torch.randn(len(label.squeeze(1)),2,100).to(device)
        # inputFeatures = torch.randn(len(label.squeeze(1)),15,100).to(device)
        output = model(points.float().to(device), features.float().to(device))
        batch_loss = criterion(output.to(device), label.squeeze(1).to(device)).to(device)
    torch.cuda.empty_cache()
    print("\n After emptying cache")
    gpu_usage()
    pTVal = pTLab.squeeze(1)
    labVal = label.squeeze(1)

    # Added distance correlation calculation between tagger output and jet pT
    outTag = f.softmax(output,dim=1)[:,1]
    normedweight = torch.ones_like(outTag)
    # disco signal parameter
    sgpVal = pT.squeeze(1).to(device)
    mask = sgpVal.gt(0).to(device)
    maskedoutTag = torch.masked_select(outTag, mask)
    maskedsgpVal = torch.masked_select(sgpVal, mask)
    maskedweight = torch.masked_select(normedweight, mask)
    batch_loss_dc = distance_corr(maskedoutTag.to(device), maskedsgpVal.to(device), maskedweight.to(device), 1).to(device)
    lambdaDC = ldc
    auc = roc_auc_score(label.to("cpu").squeeze(1).numpy(), outTag.to("cpu").detach().numpy())
    return l1*batch_loss, lambdaDC*batch_loss_dc, batch_loss_dc, auc

def main():
    rng = np.random.RandomState(2022)
    # parse arguments
    parser = ArgumentParser(config_options=MagiConfigOptions(strict = True, default="configs/C1.py"),formatter_class=ArgumentDefaultsRawHelpFormatter)
    parser.add_argument("--outf", type=str, default="logs", help='Name of folder to be used to store outputs')
    parser.add_argument("--model", type=str, default=None, help="Existing model to continue training, if applicable")
    parser.add_config_only(*c.config_schema)
    parser.add_config_only(**c.config_defaults)
    args = parser.parse_args()

    if not os.path.isdir(args.outf):
        os.mkdir(args.outf)
    # Choose cpu or gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    if device.type == 'cuda':
        gpuIndex = torch.cuda.current_device()
        print("Using GPU named: \"{}\"".format(torch.cuda.get_device_name(gpuIndex)))
        #print('Memory Usage:')
        #print('\tAllocated:', round(torch.cuda.memory_allocated(gpuIndex)/1024**3,1), 'GB')
        #print('\tCached:   ', round(torch.cuda.memory_reserved(gpuIndex)/1024**3,1), 'GB')
    torch.manual_seed(args.hyper.rseed)
    # Load dataset
    print('Loading dataset ...')
    dSet = args.dataset
    sigFiles = dSet.signal
    inputFiles = dSet.background
    hyper = args.hyper
    inputFiles.update(sigFiles)
    print(inputFiles)
    varSet = args.features.train
    inputFeatureVars = [var for var in varSet if var not in ["jCsthvCategory","jCstEvtNum","jCstJNum"]]
    print("Input feature variables:",inputFeatureVars)
    pTBins = hyper.pTBins
    uniform = args.features.uniform
    mT = args.features.mT
    weight = args.features.weight
    numConst = args.hyper.numConst
    entireDataSet = RootDataset(inputFolder=dSet.path, root_file=inputFiles, variables=varSet, pTBins=pTBins, uniform=uniform, mT=mT, weight=weight, numConst=numConst)
    randBalancedSet = splitDataSetEvenly(entireDataSet,rng,hyper.epochs)
    # Build model
    network_module = particlenet_pf
    network_options = {}
    network_options["num_of_k_nearest"] = args.hyper.num_of_k_nearest
    network_options["num_of_edgeConv_dim"] = args.hyper.num_of_edgeConv_dim
    network_options["num_of_edgeConv_convLayers"] = args.hyper.num_of_edgeConv_convLayers
    network_options["num_of_fc_layers"] = args.hyper.num_of_fc_layers
    network_options["num_of_fc_nodes"] = args.hyper.num_of_fc_nodes
    network_options["fc_dropout"] = args.hyper.fc_dropout
    model = network_module.get_model(inputFeatureVars,**network_options)
    if (args.model == None):
        #model.apply(init_weights)
        print("Creating new model ")
        args.model = 'net.pth'
    else:
        print("Loading model from " + modelLocation)
    model = copy.deepcopy(model)
    model = model.to(device)
    model.eval()
    modelLocation = "{}/{}".format(args.outf,args.model)

    # Loss function
    criterion = nn.CrossEntropyLoss()
    criterion.to(device=device)

    #Optimizer
    optimizer = optim.Adam(model.parameters(), lr = hyper.learning_rate)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.95, last_epoch=-1, verbose=True)

    # training and validation
    # writer = SummaryWriter()
    training_losses_tag = np.zeros(hyper.epochs)
    training_losses_dc = np.zeros(hyper.epochs)
    training_losses_total = np.zeros(hyper.epochs)
    validation_losses_tag = np.zeros(hyper.epochs)
    validation_losses_dc = np.zeros(hyper.epochs)
    validation_losses_total = np.zeros(hyper.epochs)
    aucs = []
    for epoch in range(hyper.epochs):
        dataset = udata.Subset(entireDataSet,randBalancedSet[epoch])
        sizes = get_sizes(len(dataset), dSet.sample_fractions)
        train, val, test = udata.random_split(dataset, sizes, generator=torch.Generator().manual_seed(42))
        loader_train = udata.DataLoader(dataset=train, batch_size=hyper.batchSize, num_workers=0, shuffle=True)
        loader_val = udata.DataLoader(dataset=val, batch_size=hyper.batchSize, num_workers=0)
        loader_test = udata.DataLoader(dataset=test, batch_size=hyper.batchSize, num_workers=0)
        print("Beginning epoch " + str(epoch))
        # training
        train_loss_tag = 0
        train_loss_dc = 0
        train_dc_val = 0
        train_loss_total = 0
        for i, data in tqdm(enumerate(loader_train), unit="batch", total=len(loader_train)):
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            batch_loss_tag, batch_loss_dc, dc_val, auc_train = processBatch(args, device, varSet, data, model, criterion, [hyper.lambdaTag, hyper.lambdaReg, hyper.lambdaGR, hyper.lambdaDC], epoch)
            aucs.append("auc train e-{} b-{}: {}\n".format(epoch,i,auc_train))
            batch_loss_total = batch_loss_tag # + batch_loss_dc
            batch_loss_total.backward()
            optimizer.step()
            model.eval()
            train_loss_tag += batch_loss_tag.item()
            #train_loss_dc += batch_loss_dc.item()
            #train_dc_val += dc_val.item()
            train_loss_total += batch_loss_total.item()
            # writer.add_scalar('training loss', train_loss_total / 1000, epoch * len(loader_train) + i)
            del batch_loss_tag, batch_loss_total, dc_val
        train_loss_tag /= len(loader_train)
        train_loss_dc /= len(loader_train)
        train_dc_val /= len(loader_train)
        train_loss_total /= len(loader_train)
        training_losses_tag[epoch] = train_loss_tag
        training_losses_dc[epoch] = train_loss_dc
        training_losses_total[epoch] = train_loss_total
        print("t_tag: "+ str(train_loss_tag))
        print("t_dc: "+ str(train_loss_dc))
        print("t_dc_val: "+ str(train_dc_val))
        print("t_total: "+ str(train_loss_total))

        # validation
        val_loss_tag = 0
        val_loss_dc = 0
        val_dc_val = 0
        val_loss_total = 0
        for i, data in enumerate(loader_val):
            output_loss_tag, output_loss_dc, dc_val, auc_val = processBatch(args, device, varSet, data, model, criterion, [hyper.lambdaTag, hyper.lambdaReg, hyper.lambdaGR, hyper.lambdaDC], epoch)
            aucs.append("auc val e-{} b-{}: {}\n".format(epoch,i,auc_val))
            output_loss_total = output_loss_tag # + output_loss_dc
            val_loss_tag += output_loss_tag.item()
            # val_loss_dc += output_loss_dc.item()
            # val_dc_val += dc_val.item()
            val_loss_total += output_loss_total.item()
            del output_loss_tag, output_loss_dc, dc_val
        val_loss_tag /= len(loader_val)
        val_loss_dc /= len(loader_val)
        val_dc_val /= len(loader_val)
        val_loss_total /= len(loader_val)
        scheduler.step()
        #scheduler.step(torch.tensor([val_loss_total]))
        validation_losses_tag[epoch] = val_loss_tag
        validation_losses_dc[epoch] = val_loss_dc
        validation_losses_total[epoch] = val_loss_total
        print("v_tag: "+ str(val_loss_tag))
        print("v_dc: "+ str(val_loss_dc))
        print("v_dc_val: "+ str(val_dc_val))
        print("v_total: "+ str(val_loss_total))
        # save the model
        model.eval()
        torch.save(model.state_dict(), modelLocation)
        torch.cuda.empty_cache()
    # writer.close()

    # plot loss/epoch for training and validation sets
    print("Making basic validation plots")
    training_tag = plt.plot(training_losses_tag, label='training_tag')
    validation_tag = plt.plot(validation_losses_tag, label='validation_tag')
    training_dc = plt.plot(training_losses_dc, label='training_dc')
    validation_dc = plt.plot(validation_losses_dc, label='validation_dc')
    training_total = plt.plot(training_losses_total, label='training_total')
    validation_total = plt.plot(validation_losses_total, label='validation_total')
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(args.outf + "/loss_plot.png")
    np.savez(args.outf + "/normMeanStd",normMean=entireDataSet.normMean,normStd=entireDataSet.normstd)
    aucFile = open(args.outf + "/auc.txt", "w+")
    aucFile.writelines(aucs)
    aucFile.close()
    parser.write_config(args, args.outf + "/config_out.py")

if __name__ == "__main__":
    main()
