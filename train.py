import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as f
from torch.utils.tensorboard import SummaryWriter
import torch.utils.data as udata
import torch.optim as optim
import os
from models import DNN, PatchLoss, WeightedPatchLoss
from dataset import RootDataset, get_sizes
import matplotlib.pyplot as plt
from magiconfig import ArgumentParser, MagiConfigOptions, ArgumentDefaultsRawHelpFormatter
from configs import configs as c
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, roc_auc_score

# ask Kevin how to create training root files for the NN
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def getNNOutput(dataset, model):
    labels, data = dataset.get_arrays()
    model.eval()
    out = model(data)
    output = f.softmax(out,dim=1)[:,0].detach().numpy()
    return labels[:,0], output

def getROCStuff(label, output):
    fpr, tpr, thresholds = roc_curve(label, output)
    auc = roc_auc_score(label, output)
    return fpr, tpr, auc

def main():
    # parse arguments
    parser = ArgumentParser(config_options=MagiConfigOptions(strict = True),formatter_class=ArgumentDefaultsRawHelpFormatter)
    parser.add_argument("--outf", type=str, default="logs", help='Name of folder to be used to store outputs')
    parser.add_argument("--batchSize", type=int, default=500, help="Training batch size")
    parser.add_argument("--model", type=str, default=None, help="Existing model to continue training, if applicable")
    parser.add_argument("--patchSize", type=int, default=20, help="Size of patches to apply in loss function")
    parser.add_argument("--kernelSize", type=int, default=3, help="Size of kernel in CNN")
    parser.add_argument("--num_of_features", type=int, default=9, help="Number of features in CNN layers")
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
        #print('Memory Usage:')
        #print('\tAllocated:', round(torch.cuda.memory_allocated(gpuIndex)/1024**3,1), 'GB')
        #print('\tCached:   ', round(torch.cuda.memory_reserved(gpuIndex)/1024**3,1), 'GB')

    # Load dataset
    print('Loading dataset ...')
    inputFiles = []
    dSet = args.dataset
    sigFiles = dSet.signal
    inputFiles = dSet.background
    hyper = args.hyper
    inputFiles.update(sigFiles)
    print(inputFiles)
    varSet = args.features.train
    print(varSet)
    dataset = RootDataset(inputFolder=dSet.path, root_file=inputFiles, variables=varSet)
    sizes = get_sizes(len(dataset), [0.70, 0.15, 0.15])
    train, val, test = udata.random_split(dataset, sizes, generator=torch.Generator().manual_seed(42))
    loader_train = udata.DataLoader(dataset=train, batch_size=args.batchSize, num_workers=0)
    loader_val = udata.DataLoader(dataset=val, batch_size=args.batchSize, num_workers=0)
    loader_test = udata.DataLoader(dataset=test, batch_size=args.batchSize, num_workers=0)

    # Build model
    model = DNN(n_var=len(varSet), n_layers=hyper.num_of_layers, n_nodes=hyper.num_of_nodes, n_outputs=2, drop_out_p=hyper.dropout).to(device=args.device)
    if (args.model == None):
        model.apply(init_weights)
        print("Creating new model ")
    else:
        print("Loading model from file " + args.model)
        model.load_state_dict(torch.load(args.model))
        model.eval()

    # Loss function
    criterion = nn.CrossEntropyLoss()
    criterion.to(device=args.device)

    #Optimizer
    optimizer = optim.Adam(model.parameters(), lr = hyper.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=0.1, patience=10, verbose=True)

    # training and validation
    writer = SummaryWriter()
    step = 0
    training_losses = np.zeros(hyper.epochs)
    validation_losses = np.zeros(hyper.epochs)
    for epoch in range(hyper.epochs):
        print("Beginning epoch " + str(epoch))
        # training
        train_loss = 0
        for i, data in enumerate(loader_train, 0):
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            label, d = data
            output = model((d.float().to(args.device)))
            batch_loss = criterion(output.to(args.device), label.squeeze(1).to(args.device)).to(args.device)
            batch_loss.backward()
            optimizer.step()
            model.eval()
            train_loss+=batch_loss.item()
            writer.add_scalar('training loss', train_loss / 1000, epoch * len(loader_train) + i)
            del label
            del d
            del output
            del batch_loss
        training_losses[epoch] = train_loss
        print("t: "+ str(train_loss))

        # validation
        val_loss = 0
        for i, data in enumerate(loader_val, 0):
            val_label, val_d =  data
            val_output = model((val_d.float().to(args.device)))
            output_loss = criterion(val_output.to(args.device), val_label.squeeze(1).to(args.device)).to(args.device)
            val_loss+=output_loss.item()
            del val_label
            del val_d
            del val_output
            del output_loss
        scheduler.step(torch.tensor([val_loss]))
        validation_losses[epoch] = val_loss
        print("v: "+ str(val_loss))

        # save the model
        model.eval()
        torch.save(model.state_dict(), os.path.join(args.outf, 'net.pth'))
    writer.close()

    # plot loss/epoch for training and validation sets
    training = plt.plot(training_losses, label='training')
    validation = plt.plot(validation_losses, label='validation')
    plt.legend()
    plt.savefig(args.outf + "/loss_plot.png")

    # plot ROC curve
    model.to('cpu')
    label_train, output_train = getNNOutput(dataset, model)
    #label_test, output_test = getNNOutput(dataset, model)
    fpr_Train, tpr_Train, auc_Train = getROCStuff(label_train, output_train)
    fig = plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve', pad=45.0)
    plt.plot(fpr_Train, tpr_Train, label="Train (area = {:.3f})".format(auc_Train))
    plt.legend(loc='best')
    fig.savefig(args.outf + "/roc_plot.pdf", dpi=fig.dpi)
    plt.close(fig)

    # plot discriminator
    bins = np.linspace(0, 1, 100)
    fig, ax = plt.subplots(figsize=(6, 6))
    y_Train_Sg = []
    y_Train_Bg = []
    for lt in range(len(label_train)):
        lbl = label_train[lt]
        if lbl == 1:
            y_Train_Sg.append(output_train[lt])
        else:
            y_Train_Bg.append(output_train[lt])
    ax.set_title('')
    ax.set_ylabel('Norm Events')
    ax.set_xlabel('Discriminator')
    plt.hist(y_Train_Sg, bins, color='xkcd:red', alpha=0.9, histtype='step', lw=2, label='Sg Train', density=True)
    plt.hist(y_Train_Bg, bins, color='xkcd:blue', alpha=0.9, histtype='step', lw=2, label='Bg Train', density=True)
    ax.legend(loc='best', frameon=False)
    fig.savefig(args.outf + "/discriminator.pdf", dpi=fig.dpi)

if __name__ == "__main__":
    main()
