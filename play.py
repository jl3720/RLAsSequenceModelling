import numpy as np
import enum
import copy
import connect4.connect4 as game
from pympler import asizeof
import deeplearning.buffer as buf
import torch 
import torch.nn as nn
import torch.optim as optim
import deeplearning.mlp as mlp
import torch.nn.functional as F
from deeplearning.league import League
import matplotlib.pyplot as plt

lea = League()

while True:
    lea.play_season()