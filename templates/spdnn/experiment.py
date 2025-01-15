import argparse
import json
import os
import pathlib
import time
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

### SPDNN code imports ###
import torch.nn as nn
from torch.autograd import Function
import numpy as np
#PhotonActivation.py file code#
class PhotonCountingP(nn.Module):
    """ The probability of 1 photon in photon counting 
        (also the expectation value) with mean flux x """
    def __init__(self):
        super(PhotonCountingP, self).__init__()

    def forward(self, x):
        return 1.-torch.exp(torch.abs(x)*-1.)
    
class BernoulliFunctionST(Function):
    """ The 'Straight Through' stochastic Bernoulli activation"""
    @staticmethod
    def forward(ctx, input):

        return torch.bernoulli(input)

    @staticmethod
    def backward(ctx, grad_output):

        return grad_output

class PoissonFunctionST(Function):
    """ The 'Straight Through' stochastic Poisson activation"""
    @staticmethod
    def forward(ctx, input):

        return torch.poisson(input)

    @staticmethod
    def backward(ctx, grad_output):

        return grad_output
    
PoissonST = PoissonFunctionST.apply    
BernoulliST = BernoulliFunctionST.apply   

class PhotonActivation(nn.Module):

    def __init__(self,sampler='bernoulli'):
        super(PhotonActivation, self).__init__()
        self.act = PhotonCountingP()
        if sampler == 'poisson':
            self.sampler = PoissonST
        elif sampler == 'bernoulli':
            self.sampler = BernoulliST
        else:
            raise

    def forward(self, input, n_rep=1, slope=1.):
        x = input
        probs = self.act(slope * x)
        out = self.sampler(probs)
        if self.sampler == BernoulliST:
            probs = self.act(x)
        elif self.sampler == PoissonST:
            probs = torch.abs(x)
        else: raise
        if n_rep==0:  # Infinite number of shots per activation
            out = probs
        else:
            out = self.sampler(probs.unsqueeze(0).repeat((n_rep,)+(1,)*len(probs.shape))).mean(axis=0)*torch.sign(x)
        return out
        out = self.sampler(probs)
        return out

class PhotonActivationCoh(nn.Module):

    def __init__(self,sampler='bernoulli'):
        super(PhotonActivationCoh, self).__init__()
        self.act = PhotonCountingP()
        if sampler == 'poisson':
            self.sampler = PoissonST
        elif sampler == 'bernoulli':
            self.sampler = BernoulliST
        else:
            raise

    def forward(self, input, n_rep=1, slope=1.):
        x = input**2
        probs = self.act(slope * x)
        out = self.sampler(probs)
        if self.sampler == BernoulliST:
            probs = self.act(x)
        elif self.sampler == PoissonST:
            probs = torch.abs(x)
        else: raise
        if n_rep==0:  # Infinite number of shots per activation
            out = probs
        else:
            out = self.sampler(probs.unsqueeze(0).repeat((n_rep,)+(1,)*len(probs.shape))).mean(axis=0)*torch.sign(x)
        return out
#End of PhotonActivation.py File#

#Start of SPDNN_CONV.py#
def PDConv(n_in=128, n_out=128, s=1, ks=3, batchnorm=True):
    if batchnorm:
        return [
            nn.Conv2d(in_channels=n_in, out_channels=n_out, kernel_size=ks, stride=s, padding=int((ks-1)/2*s),bias=False),
            nn.BatchNorm2d(n_out),
            PhotonActivationCoh()
        ]
    else:
        return [
            nn.Conv2d(in_channels=n_in, out_channels=n_out, kernel_size=ks, stride=s, padding=int((ks-1)/2*s),bias=False),
            PhotonActivationCoh()
        ]
    
def PDConvsAP(n_in=3, n_chan=[128,128], ss=[1,1], kss=[3,3], batchnorm=True):
    modules = []
    n_list = [n_in]+list(n_chan)
    for i in range(len(n_chan)):
        modules += PDConv(n_in=n_list[i],n_out=n_list[i+1],s=ss[i],ks=kss[i],batchnorm=batchnorm)\
                      +[nn.AvgPool2d((2,2))]

    return nn.Sequential(*modules)

class PDConvNet(nn.Module):
    
    def __init__(self, n_linear=100, n_output=10, d_input=(1,28,28), n_chan=[128,128], ss=[1,1], kss=[3,3], batchnorm=True, dropout=None, last_layer_bias=True, linear_act='PD', sampler='bernoulli'):
        super(PDConvNet, self).__init__()
        
        self.sampler = sampler
        self.n_chan = n_chan
        self.batchnorm = batchnorm
        
        self.d_input = d_input
        self.n_output = n_output
        self.n_linear = n_linear

        self.convs = PDConvsAP(n_in=d_input[0],n_chan=n_chan,ss=ss,kss=kss,batchnorm=batchnorm)
        self.flat = nn.Flatten()
        
        self.fc1 = nn.Linear(int(n_chan[-1]*(d_input[1]//2**len(n_chan)//np.prod(ss))**2), n_linear, bias=False)
        if linear_act=='PD':
            self.linear_act = PhotonActivationCoh(sampler=sampler)
        elif linear_act=='ReLU':
            self.linear_act = nn.ReLU()
        self.fc2 = nn.Linear(n_linear, n_output, bias=last_layer_bias)
        if dropout is None:
            self.dropout = None
        else:
            self.dropout = nn.Dropout(dropout)

    def forward(self, x, n_rep=1, slope=1.):
        x = x.view(-1, *self.d_input)
        for layer in self.convs:
            if isinstance(layer, PhotonActivationCoh):
                x = layer(x, n_rep=n_rep, slope=slope)
            else:
                x = layer(x)
        x = self.flat(x)
        x = self.fc1(x)
        if isinstance(self.linear_act, PhotonActivationCoh):
            x = self.linear_act(x, n_rep=n_rep, slope=slope)
        else:
            x = self.linear_act(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.fc2(x)
        x_out = F.log_softmax(x, dim=1)
        return x_out
#End of SPDNN_Conv.py#

#Start of SPDNN_MLP.py#
class PDMLP_1(nn.Module):
    
    def __init__(self, n_hidden=100, n_input=784, n_output=10, dropout=None, last_layer_bias=False, sampler='bernoulli'):
        super(PDMLP_1, self).__init__()
        
        self.sampler = sampler
        
        self.n_input = n_input
        self.n_output = n_output
        self.n_hidden = n_hidden

        self.fc1 = nn.Linear(n_input, n_hidden, bias=False)
        self.act = PhotonActivation(sampler=sampler)
        self.fc2 = nn.Linear(n_hidden, n_output, bias=last_layer_bias)
        if dropout is None:
            self.dropout = None
        else:
            self.dropout = nn.Dropout(dropout)

    def forward(self, x, n_rep=1, slope=1.):
        x = x.view(-1, self.n_input)
        x_fc1 = self.act(self.fc1(x), n_rep=n_rep, slope=slope)
        if self.dropout is not None:
            x_fc1 = self.dropout(x_fc1)
        x_fc2 = self.fc2(x_fc1)
        x_out = F.log_softmax(x_fc2, dim=1)
        return x_out

class incoh_PDMLP(nn.Module):
    
    def __init__(self, n_hiddens=[100,100], n_input=784, n_output=10, sampler='bernoulli',output_bias=True):
        super(incoh_PDMLP, self).__init__()
        
        self.sampler = sampler
        
        self.n_input = n_input
        self.n_output = n_output
        self.n_hiddens = n_hiddens
        
        n_nodes = [n_input]+list(n_hiddens)
        self.fcs = nn.ModuleList([nn.Linear(i,j,bias=False) for i, j in zip(n_nodes[:-1], n_nodes[1:])])
        self.last_fc = nn.Linear(n_hiddens[-1],n_output,bias=output_bias)
        self.act = PhotonActivation(sampler=sampler)

    def forward(self, x, n_rep=1, slope=1.):
        x = x.view(-1, self.n_input)
        for fc in self.fcs:
            x = fc(x)
            x = self.act(x, n_rep=n_rep, slope=slope)
        x_out = F.log_softmax(self.last_fc(x), dim=1)
        return x_out
    
class coh_PDMLP(nn.Module):
    
    def __init__(self, n_hiddens=[100,100], n_input=784, n_output=10, sampler='bernoulli',output_bias=True):
        super(coh_PDMLP, self).__init__()
        
        self.sampler = sampler
        
        self.n_input = n_input
        self.n_output = n_output
        self.n_hiddens = n_hiddens
        
        n_nodes = [n_input]+list(n_hiddens)
        self.fcs = nn.ModuleList([nn.Linear(i,j,bias=False) for i, j in zip(n_nodes[:-1], n_nodes[1:])])
        self.last_fc = nn.Linear(n_hiddens[-1],n_output,bias=output_bias)
        self.act = PhotonActivationCoh(sampler=sampler)

    def forward(self, x, n_rep=1, slope=1.):
        x = x.view(-1, self.n_input)
        for fc in self.fcs:
            x = fc(x)
            x = self.act(x, n_rep=n_rep, slope=slope)
        x_out = F.log_softmax(self.last_fc(x), dim=1)
        return x_out
#End of SPDNN_MLP.py#


# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training Function
def train(model, train_loader, optimizer, epoch, slope=1.0):
    model.train()
    train_loss, correct = 0.0, 0
    print(f'\n# Epoch {epoch} - Slope {slope}')
    
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        with autocast():  # Enable mixed precision
            output = model(data)
            loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    
    train_loss /= len(train_loader.dataset)
    train_acc = correct / len(train_loader.dataset)
    print(f"Training Loss: {train_loss:.4f}, Accuracy: {train_acc*100:.2f}%")
    return train_loss, train_acc

# Testing Function
def test(model, test_loader):
    model.eval()
    test_loss, correct = 0.0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    test_acc = correct / len(test_loader.dataset)
    print(f"Test Loss: {test_loss:.4f}, Accuracy: {test_acc*100:.2f}%")
    return test_loss, test_acc

# Main Experiment Function
def main(config):
    # Setup output directories
    pathlib.Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)
    os.makedirs(os.path.join(config["output_dir"], "results"), exist_ok=True)

    # Dataset
    transform = transforms.Compose([
        transforms.Resize(config["image_size"]),
        transforms.ToTensor(),
    ])
    train_loader = DataLoader(
        datasets.MNIST(config["data_dir"], train=True, download=True, transform=transform),
        batch_size=config["batch_size"], shuffle=True
    )
    test_loader = DataLoader(
        datasets.MNIST(config["data_dir"], train=False, transform=transform),
        batch_size=config["batch_size"], shuffle=False
    )

    # Model Selection
    if config["model_type"] == "conv":
        model = PDConvNet(
            d_input=(1, config["image_size"], config["image_size"]),
            n_linear=config["n_linear"],
            n_output=10,
            n_chan=config["n_channels"],
            kss=config["kernel_sizes"],
            ss=config["strides"],
            batchnorm=False
        ).to(device)
        model_type = "ConvNN"
    elif config["model_type"] == "mlp":
        model = PDMLP_1(
            n_hidden=config["n_linear"],
            n_input=config["image_size"]**2,
            n_output=10
        ).to(device)
        model_type = "MLP"
    else:
        raise ValueError("Invalid model_type! Choose 'conv' or 'mlp'.")

    optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"])

    # Train and Test
    start_time = time.time()
    best_acc = 0.0
    train_acc_track, test_acc_track = [], []

    for epoch in range(1, config["epochs"] + 1):
        train_loss, train_acc = train(model, train_loader, optimizer, epoch)
        test_loss, test_acc = test(model, test_loader)
        best_acc = max(best_acc, test_acc)

        # Track results
        train_acc_track.append(train_acc)
        test_acc_track.append(test_acc)

    # Save Results and Model with Explicit Naming
    if model_type == "ConvNN":
        model_name = f"MNIST_{config['image_size']}x{config['image_size']}_ConvNN_N{config['n_linear']}_SGD_lr{config['lr']}_mom{config['momentum']}"
    elif model_type == "MLP":
        model_name = f"MNIST_{config['image_size']}x{config['image_size']}_MLP_N{config['n_linear']}_SGD_lr{config['lr']}_mom{config['momentum']}"

    final_info = {
    "training_time": {"means": round(time.time() - start_time, 2)},
    "train_accuracy": {"means": train_acc_track},
    "test_accuracy": {"means": test_acc_track},
    "best_accuracy": {"means": round(best_acc, 4)},
    "eval_loss": {"means": round(test_loss, 4)}
    }
    # Use the correct output directory
    final_info_path = os.path.join(config["output_dir"], "final_info.json")
    with open(final_info_path, "w") as f:
        json.dump(final_info, f, indent=4)

    # Save model checkpoint
    model_path = os.path.join(config["output_dir"], f"{model_name}.pth")
    torch.save(model.state_dict(), model_path)

    print(f"Training complete. Results saved to {final_info_path}")
    print(f"Model saved to {model_path}")



# Argument Parsing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SPDNN Experiment")
    parser.add_argument("--data_dir", type=str, default="./data", help="Path to dataset")
    parser.add_argument("--output_dir", type=str, default="run_0", help="Output directory")
    parser.add_argument("--image_size", type=int, default=28, help="Image size for MNIST")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for optimizer")
    parser.add_argument("--n_linear", type=int, default=400, help="Number of neurons in linear layer")
    parser.add_argument("--n_channels", type=int, nargs='+', default=[16], help="Channels for conv layers")
    parser.add_argument("--kernel_sizes", type=int, nargs='+', default=[5], help="Kernel sizes for conv layers")
    parser.add_argument("--strides", type=int, nargs='+', default=[1], help="Strides for conv layers")
    parser.add_argument("--model_type", type=str, choices=["conv", "mlp"], default="conv",
                        help="Model type: 'conv' for ConvNN, 'mlp' for MLP.")

    args = parser.parse_args()
    config = vars(args)
    print("Experiment Configuration:", config)
    main(config)

