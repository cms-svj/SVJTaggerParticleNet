"""
Adapted from https://github.com/SaoYan/DnCNN-PyTorch/blob/master/models.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.autograd import Variable

class DnCNN(nn.Module):
    def __init__(self, channels=1, num_of_layers=9, kernel_size=3, features=100):
        super(DnCNN, self).__init__()
        padding = int((kernel_size-1)/2)
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))            
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.dncnn(x)
        return out

class DNN(nn.Module):
    def __init__(self, n_var=10, n_layers=1, n_nodes=20, n_outputs=2, drop_out_p=0.3):
        super(DNN, self).__init__()
        layers = []
        layers.append(nn.Linear(n_var, n_nodes))
        layers.append(nn.ReLU())

        for n in list(n_nodes for x in range(n_layers)):
            layers.append(nn.Linear(n, n))
            layers.append(nn.ReLU())

        layers.append(nn.Dropout(p=drop_out_p))
        layers.append(nn.Linear(n_nodes, n_outputs))

        self.dnn = nn.Sequential(*layers)

    def forward(self, x):
        return self.dnn(x)

class PatchLoss(nn.Module):
    def __initII(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(PatchLoss, self).__init__(size_average, reduce, reduction)

    def forward(self, output, target, patch_size=50):
        avg_loss = 0
        for i in range(len(output)):
            # split output and target images into patches
            output_patches = output[i].unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
            target_patches = target[i].unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
            max_patch_loss = 0
            # calculate loss for each patch of the image
            for j in range(list(output_patches.size())[0]):
                for k in range(list(output_patches.size())[1]):
                    max_patch_loss = max(max_patch_loss, f.l1_loss(output_patches[j][k], target_patches[j][k]))
            avg_loss+=max_patch_loss
        avg_loss/=len(output)       
        #print(avg_loss)
        return avg_loss;

class WeightedPatchLoss(nn.Module):
    def __initII(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(PatchLoss, self).__init__(size_average, reduce, reduction)

    def forward(self, output, target, patch_size):
        avg_loss = 0
        for i in range(len(output)):
            # split output and target images into patches
            output_patches = output[i].unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
            target_patches = target[i].unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
            weighted_loss = 0
            # calculate loss for each patch of the image
            for j in range(list(output_patches.size())[0]):
                for k in range(list(output_patches.size())[1]):
                    weighted_loss += f.l1_loss(output_patches[j][k],target_patches[j][k]) * torch.mean(target_patches[j][k])
            avg_loss+=weighted_loss/torch.mean(target[i])
        avg_loss/=len(output)
        return avg_loss;


if __name__=="__main__":
    # Test CrossEntropyLoss function
    # input is 5 events with 2 outputs
    # target is 5 numbers that correspond to the correct index number for each event
    # Note can not have softmax in model since CrossEntropyLoss does this tep for you. Will need to apply softmax when using the model after training. 
    loss = nn.CrossEntropyLoss()
    input = torch.randn(5, 2, requires_grad=True)
    target = torch.empty(5, dtype=torch.long).random_(2)
    output = loss(input, target)
    output.backward()

    # Test DNN model
    # input is 5 events with 10 input variables
    # out is the non-softmax output of the model
    # outFinal is the softmax final version of the model output
    # target is 5 numbers that correspond to the correct index number for each event    
    net = DNN()
    loss = nn.CrossEntropyLoss()
    input = torch.randn(5, 10)
    out = net(input)
    outFinal = f.softmax(out,dim=1)
    target = torch.empty(5, dtype=torch.long).random_(2)
    output = loss(out, target)
    output.backward()

