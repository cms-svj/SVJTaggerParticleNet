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
from dataset import RootDataset, get_sizes
import matplotlib.pyplot as plt
from magiconfig import ArgumentParser, MagiConfigOptions, ArgumentDefaultsRawHelpFormatter
from configs import configs as c
import numpy as np
from tqdm import tqdm
from Disco import distance_corr
import copy
from GPUtil import showUtilization as gpu_usage

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["NCCL_DEBUG"] = "INFO"

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def processBatch(args, device, data, model, criterion, lambdas, epoch):
    label, points, features, jetFeatures, inputFileIndex, pT, w, med, dark, rinv, alpha = data
    l1, l2, lgr, ldc = lambdas
    #print("\n Initial GPU Usage")
    #gpu_usage()
    with autocast():
        # inputPoints = torch.randn(len(label.squeeze(1)),2,100).to(device)
        # inputFeatures = torch.randn(len(label.squeeze(1)),15,100).to(device)
        output = model(points.float().to(device), features.float().to(device))
        if torch.isnan(torch.sum(output)):
            print("output has nan:", output)
        batch_loss = criterion(output.to(device), label.squeeze(1).to(device)).to(device)
    torch.cuda.empty_cache()
    #print("\n After emptying cache")
    #gpu_usage()
    labVal = label.squeeze(1)

    # Added distance correlation calculation between tagger output and jet pT
    outSoftmax = f.softmax(output,dim=1)
    signalIndex = args.hyper.num_classes - 1
    outTag = outSoftmax[:,signalIndex]
    normedweight = torch.ones_like(outTag)
    # disco signal parameter
    #sgpVal = pT.squeeze(1).to(device)
    #mask = sgpVal.gt(signalIndex-1).to(device)
    #maskedoutTag = torch.masked_select(outTag, mask)
    #maskedsgpVal = torch.masked_select(sgpVal, mask)
    #maskedweight = torch.masked_select(normedweight, mask)
    #batch_loss_dc = distance_corr(maskedoutTag.to(device), maskedsgpVal.to(device), maskedweight.to(device), 1).to(device)
    #lambdaDC = ldc
    return l1*batch_loss #, lambdaDC*batch_loss_dc, batch_loss_dc

def main():
    rng = np.random.RandomState(2022)
    # parse arguments
    parser = ArgumentParser(config_options=MagiConfigOptions(strict = True, default="configs/C1.py"),formatter_class=ArgumentDefaultsRawHelpFormatter)
    parser.add_argument("--outf", type=str, default="logs", help='Name of folder to be used to store outputs')
    parser.add_argument("--inpf", type=str, default="processedData_nc100", help='Name of the npz input training file')
    parser.add_argument("--model", type=str, default=None, help="Existing model to continue training, if applicable")
    parser.add_config_only(*c.config_schema)
    parser.add_config_only(**c.config_defaults)
    args = parser.parse_args()
    if not os.path.isdir(args.outf):
        os.mkdir(args.outf)
    parser.write_config(args, args.outf + "/config_out.py")
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
    trainNPZ = "processedDataNPZ/processedData_nc100_train_uniformPt.npz"
    valNPZ = "processedDataNPZ/processedData_nc100_validation_uniformPt.npz"
    train = RootDataset(trainNPZ)
    val = RootDataset(valNPZ)
    inputFeatureVars = train.inputFeaturesVarName
    print("Input jet constituent features:",inputFeatureVars)
    loader_train = udata.DataLoader(dataset=train, batch_size=hyper.batchSize, num_workers=0, shuffle=True)
    loader_val = udata.DataLoader(dataset=val, batch_size=hyper.batchSize, num_workers=0, shuffle=False)
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
    if (args.model == None):
        #model.apply(init_weights)
        print("Creating new model ")
        args.model = 'net.pth'
    else:
        print("Loading model from " + modelLocation)
    model = copy.deepcopy(model)
    model = model.to(device)
    model.eval()
    modelInfo = []
    modelInfo.append("Model contains {} trainable parameters.".format(count_parameters(model)))
    with open('{}/modelInfo.txt'.format(args.outf), 'w') as f:
        for line in modelInfo:
            f.write("{}\n".format(line))
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
    for epoch in range(hyper.epochs):
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
            batch_loss_tag = processBatch(args, device, data, model, criterion, [hyper.lambdaTag, hyper.lambdaReg, hyper.lambdaGR, hyper.lambdaDC], epoch)
            batch_loss_total = batch_loss_tag # + batch_loss_dc
            batch_loss_total.backward()
            optimizer.step()
            model.eval()
            train_loss_tag += batch_loss_tag.item()
            #train_loss_dc += batch_loss_dc.item()
            #train_dc_val += dc_val.item()
            train_loss_total += batch_loss_total.item()
            # writer.add_scalar('training loss', train_loss_total / 1000, epoch * len(loader_train) + i)
        train_loss_tag /= len(loader_train)
        #train_loss_dc /= len(loader_train)
        #train_dc_val /= len(loader_train)
        train_loss_total /= len(loader_train)
        training_losses_tag[epoch] = train_loss_tag
        #training_losses_dc[epoch] = train_loss_dc
        training_losses_total[epoch] = train_loss_total
        if np.isnan(train_loss_tag):
            print("nan in training")
            break
        print("t_tag: "+ str(train_loss_tag))
        #print("t_dc: "+ str(train_loss_dc))
        #print("t_dc_val: "+ str(train_dc_val))
        print("t_total: "+ str(train_loss_total))

        # validation
        val_loss_tag = 0
        val_loss_dc = 0
        val_dc_val = 0
        val_loss_total = 0
        for i, data in enumerate(loader_val):
            output_loss_tag = processBatch(args, device, data, model, criterion, [hyper.lambdaTag, hyper.lambdaReg, hyper.lambdaGR, hyper.lambdaDC], epoch)
            output_loss_total = output_loss_tag # + output_loss_dc
            val_loss_tag += output_loss_tag.item()
            # val_loss_dc += output_loss_dc.item()
            # val_dc_val += dc_val.item()
            val_loss_total += output_loss_total.item()
        val_loss_tag /= len(loader_val)
        #val_loss_dc /= len(loader_val)
        #val_dc_val /= len(loader_val)
        val_loss_total /= len(loader_val)
        scheduler.step()
        #scheduler.step(torch.tensor([val_loss_total]))
        validation_losses_tag[epoch] = val_loss_tag
        #validation_losses_dc[epoch] = val_loss_dc
        validation_losses_total[epoch] = val_loss_total
        if np.isnan(val_loss_tag):
            print("nan in val")
            break
        print("v_tag: "+ str(val_loss_tag))
        #print("v_dc: "+ str(val_loss_dc))
        #print("v_dc_val: "+ str(val_dc_val))
        print("v_total: "+ str(val_loss_total))
        # save the model
        model.eval()
        modelLocation = "{}/net_{}.pth".format(args.outf,epoch)
        torch.save(model.state_dict(), modelLocation)
        torch.cuda.empty_cache()
    # writer.close()

    # plot loss/epoch for training and validation sets
    print("Making basic validation plots")
    #training_tag = plt.plot(training_losses_tag, label='training_tag')
    #validation_tag = plt.plot(validation_losses_tag, label='validation_tag')
    #training_dc = plt.plot(training_losses_dc, label='training_dc')
    #validation_dc = plt.plot(validation_losses_dc, label='validation_dc')
    training_total = plt.plot(training_losses_total, label='training_total')
    validation_total = plt.plot(validation_losses_total, label='validation_total')
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(args.outf + "/loss_plot.png")

if __name__ == "__main__":
    main()
