##Main code taken from Udacity's repository:
##https://github.com/udacity/deep-reinforcement-learning/
##

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


##Hidden layer's weights intialization
def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=400, fc2_units=300):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        
        _128_unit = 128
        _256_unit = 256
        _512_unit = 512
        self.seed = torch.manual_seed(seed)
        
        # RNN Layer
        self.step_window = 1
        self.hidden_dim = 32
        n_layers = 1
        #self.input = nn.RNN(state_size, self.hidden_dim, n_layers, batch_first=True)
        
        #self.fc1 = nn.Linear(state_size, _256_unit)
        self.fc2 = nn.Linear(state_size * self.step_window, _256_unit)
        #self.fc3 = nn.Linear(_512_unit, _512_unit)
        self.fc4 = nn.Linear(_256_unit, _128_unit)
        self.fc5 = nn.Linear(_128_unit, action_size)
        
        #batchnorm
        self.bn128 = nn.BatchNorm1d(_128_unit)
        self.bn256 = nn.BatchNorm1d(_256_unit)
        self.bn512 = nn.BatchNorm1d(_512_unit)

        #dropout
        self.dpout = nn.Dropout(p=0.33)
        self.reset_parameters()

    def reset_parameters(self):
        #self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        #self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(*hidden_init(self.fc4))
        self.fc5.weight.data.uniform_(-3e-3, 3e-3)
        
    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        #Input layer
#         x, h = self.input(state)
#         x = x.contiguous().view(state.shape[0], -1)
#         x = F.relu(x)
#         ###x = F.relu(self.fc1(state))
        
        #First hidden layer Dense+Batchnorm+Relu
        x = self.fc2(state)
        x = self.bn256(x)
        x = F.relu(self.dpout(x))
        
        #Second hidden layer Dense+Batchnorm+Relu
#         x = self.fc3(x)
#         x = self.bn512(x)
#         x = F.relu(self.dpout(x))
        
        #Third hidden layer Dense+Batchnorm+Relu
        x = self.fc4(x)
#         x = self.bn128(x)
        x = F.relu(x)
        
        #Output layer
        out = F.tanh(self.fc5(x))
        return out


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fcs1_units=400, fc2_units=300):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        _128_unit = 128
        _256_unit = 256
        _512_unit = 512
        self.seed = torch.manual_seed(seed)
        # RNN Layer
        self.step_window = 1
        self.hidden_dim = 32
        n_layers = 1
        #self.input = nn.RNN(state_size, self.hidden_dim, n_layers, batch_first=True)

        #self.fcs1 = nn.Linear(state_size, _256_unit)
        self.fc2 = nn.Linear(self.step_window * state_size + action_size, _256_unit)
        #self.fc3 = nn.Linear(_512_unit, _512_unit)
        self.fc4 = nn.Linear(_256_unit, _128_unit)
        self.fc5 = nn.Linear(_128_unit, action_size)
        
        #batchnorm
        self.bn128 = nn.BatchNorm1d(_128_unit)
        self.bn256 = nn.BatchNorm1d(_256_unit)
        self.bn512 = nn.BatchNorm1d(_512_unit)
        
        #Dropout
        self.dpout = nn.Dropout(p=0.33)

        """
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()
        """
        
    def reset_parameters(self):
        #self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        #self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(*hidden_init(self.fc4))
        self.fc5.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
      
        #Input layer
#         x, h = self.input(state)
#         x = x.contiguous().view(state.shape[0], -1)
#         x = F.relu(x)
        #xs = F.relu(self.fcs1(state))
        
        #Concat state & action
        x = torch.cat((state, action), dim=1)

        #First hidden layer Dense+Batchnorm+Relu
        x = self.fc2(x) 
        x = self.bn256(x)
        x = F.relu(self.dpout(x))
        
        #Second hidden layer Dense+Batchnorm+Relu
#         x = self.fc3(x)
#         x = self.bn512(x)
#         x = F.relu(self.dpout(x))
        
        #Third hidden layer Dense+Batchnorm+Relu
        x = self.fc4(x)
#         x = self.bn128(x)
        x = F.relu(x)
        
        #Output layer
        out = self.fc5(x)
        return out

    
        """
        xs = F.relu(self.fcs1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        
        return self.fc3(x)
        """