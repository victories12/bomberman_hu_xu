import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import settings as s
from .agent_settings import NUM_CHANNELS,NUM_ACTIONS

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride=1, padding=0):
        super(DepthwiseSeparableConv, self).__init__()
        
        # Depth-wise convolution
        self.depthwise = nn.Conv2d(input_channels, input_channels, kernel_size, stride, padding, groups=input_channels)
        
        # Point-wise convolution
        self.pointwise = nn.Conv2d(input_channels, output_channels, 1)  # 1x1 convolution

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class BombermanNet(nn.Module):
    def __init__(self, input_dim=6, output_dim=6):  
        super(BombermanNet, self).__init__()  # Initialization of the base class
        
        # Define the CNN
        # self.conv1 = nn.Conv2d(input_dim, 32, kernel_size=5, stride=1, padding=1)
        # self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=1)
        self.conv1 = DepthwiseSeparableConv(input_dim, 32, kernel_size=5, stride=1, padding=1)
        self.conv2 = DepthwiseSeparableConv(32, 64, kernel_size=5, stride=1, padding=1)

        self.pool = nn.MaxPool2d(5, 5)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)

        self.dropout1 = nn.Dropout(0.1)
        # self.dropout2 = nn.Dropout(0.1)
        
        self.fc1 = nn.Linear(256, 32)  
        self.fc2 = nn.Linear(32, output_dim)
    
    def forward(self, x):

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        # Flatten the tensor
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)

        return x


def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    
    # Gather information about the game state
    arena = game_state['field']
    coins = game_state['coins']
    _, score, bombs_left, (x, y) = game_state['self']
    others = [xy for (n, s, b, xy) in game_state['others']]
    explosion_map = game_state['explosion_map']
    bombs = game_state['bombs']
    
    # For example, you could construct several channels of equal shape, ...
    channels = np.zeros((NUM_CHANNELS, s.COLS, s.ROWS))

    
    # Crates channel
    channels[0] = np.where(arena == 1, 1, 0)

    # Coins channel
    for xy in coins:
        channels[1][xy[1]][xy[0]] = 1

    # Player channel
    channels[2][y][x] = 1 

    # Opponents channel (using static method)
    for xy in others:
        channels[3][xy[1]][xy[0]] = 1

    # Explosion channel
    channels[4] = explosion_map

    # Bombs channel
    for xy,t in bombs:
        channels[5][xy[1]][xy[0]] = 1
        channels[6][xy[1]][xy[0]] = t

    # Environment channel (wall)
    channels[7] = np.where(arena == -1, 1, 0)

    return torch.tensor(channels, dtype=torch.float32)