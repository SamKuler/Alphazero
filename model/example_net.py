import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from functools import reduce

from env.base_env import BaseGame

class BaseNetConfig:
    def __init__(
        self, 
        num_channels:int = 256,
        dropout:float = 0.3,
        linear_hidden:list[int] = [256, 128],
    ):
        self.num_channels = num_channels
        self.linear_hidden = linear_hidden
        self.dropout = dropout
        
class MLPNet(nn.Module):
    def __init__(self, observation_size:tuple[int, int], action_space_size:int, config:BaseNetConfig, device:torch.device='cpu'):
        super().__init__()
        self.config = config
        self.device = device
        input_dim = observation_size[0] * observation_size[1] if len(observation_size) == 2 else observation_size[0]
        self.layer1 = nn.Linear(input_dim, config.linear_hidden[0])
        self.layer2 = nn.Linear(config.linear_hidden[0], config.linear_hidden[1])
        
        self.policy_head = nn.Linear(config.linear_hidden[1], action_space_size)
        self.value_head = nn.Linear(config.linear_hidden[1], 1)
        self.relu = nn.ReLU()
        self.to(device)

    def forward(self, x: torch.Tensor):
        #                                                         x: batch_size x board_x x board_y
        x = x.view(x.size(0), -1) # reshape tensor to 1d vectors, x.size(0) is batch size
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        pi = self.policy_head(x)
        v = self.value_head(x)
        return F.log_softmax(pi, dim=1), torch.tanh(v)


class LinearModel(nn.Module):
    def __init__(self, observation_size:tuple[int, int], action_space_size:int, config:BaseNetConfig, device:torch.device='cpu'):
        super(LinearModel, self).__init__()
        
        self.action_size = action_space_size
        self.config = config
        self.device = device
        
        observation_size = reduce(lambda x, y: x*y , observation_size, 1)
        self.l_pi = nn.Linear(observation_size, action_space_size)
        self.l_v  = nn.Linear(observation_size, 1)
        self.to(device)
    
    def forward(self, s: torch.Tensor):
        #                                                           s: batch_size x board_x x board_y
        s = s.view(s.shape[0], -1)                                # s: batch_size x (board_x * board_y)
        pi = self.l_pi(s)
        v = self.l_v(s)
        return F.log_softmax(pi, dim=1), torch.tanh(v)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )
    
    def forward(self, x):
        x = x + self.block(x)
        return nn.ReLU()(x)

class MyNet(nn.Module):
    def __init__(self, observation_size:tuple[int, int], action_space_size:int, config:BaseNetConfig, device:torch.device='cpu'):
        super().__init__()
        self.config = config
        self.device = device
        ########################
        # TODO: your code here #
        assert len(observation_size)==2, "In board mode only!"
        num_channels = getattr(config, 'num_channels', 64)
        dropout = getattr(config, 'dropout', 0.2)
        linear_hidden = getattr(config, 'linear_hidden', [128, 128])

        # Input
        self.conv_in = nn.Sequential(
            nn.Conv2d(1, num_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True),
        )

        self.res_layers = nn.Sequential(
            *[ResidualBlock(num_channels) for _ in range(4)]
        )
        self.dropout = nn.Dropout(dropout)

        # Flatten
        flatten_dim = num_channels * observation_size[0] * observation_size[1]

        # Policy
        self.policy_head = nn.Sequential(
            nn.Linear(flatten_dim, linear_hidden[0]),
            nn.ReLU(inplace=True),
            nn.Linear(linear_hidden[0], action_space_size)
        )
        # Value
        self.value_head = nn.Sequential(
            nn.Linear(flatten_dim, linear_hidden[0]),
            nn.ReLU(inplace=True),
            nn.Linear(linear_hidden[0], 1)
        )
        self.to(device)

        ########################
    
    def forward(self, s: torch.Tensor):
        ########################
        # TODO: your code here #
        # s: batch_size x board_x x board_y
        if s.dim() == 3:
            s = s.unsqueeze(1)

        s = self.conv_in(s)
        s = self.res_layers(s)
        s = self.dropout(s)

        s = s.view(s.size(0), -1)

        pi = self.policy_head(s)
        v = self.value_head(s)

        return F.log_softmax(pi, dim=1), torch.tanh(v)
        ########################