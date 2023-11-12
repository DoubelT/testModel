import torch
import torch.nn as nn
from collections import OrderedDict

class BankNodes(nn.Module):
    def __init__(self, config, is_training=True):
        super(BankNodes, self).__init__()
        self.config = config
        self.training = is_training

        # Define the layers using OrderedDict
        layers = OrderedDict()
        layers['fc1'] = nn.Linear(in_features=config["input_size"], out_features=config["hiddenlayer_size"])
        layers['relu1'] = nn.ReLU()
        layers['fc2'] = nn.Linear(in_features=config["hiddenlayer_size"], out_features=config["outputlayer_size"])
        layers['sig2'] = nn.Hardsigmoid()

        # Create the neural network using the layers
        self.network = nn.Sequential(layers)

    def forward(self, x):
        # Forward pass through the network
        return self.network(x)