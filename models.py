import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.autograd import Variable

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

