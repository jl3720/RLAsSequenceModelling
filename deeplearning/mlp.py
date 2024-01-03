# MLP UDPRL stolen from https://github.com/BY571/Upside-Down-Reinforcement-Learning/blob/master/Upside-Down.ipynb

import torch
from torch import nn
import numpy as np
from connect4.connect4 import Player

class BF(Player, nn.Module):
    def __init__(self, state_space, action_space, hidden_size, seed, device):
        super(BF, self).__init__()
        torch.manual_seed(seed)
        self.device = device
        self.actions = np.arange(action_space)
        self.action_space = action_space
        self.fc1 = nn.Linear(state_space, hidden_size)
        self.commands = nn.Linear(2, hidden_size)
        self.layers = 5
        self.fc = [nn.Linear(hidden_size, hidden_size).to(self.device) for x in range(0, self.layers)]
        self.last = nn.Linear(hidden_size, action_space)

        self.sigmoid = nn.Sigmoid()

    def step(self, state, desire, elo):
        return self.action(torch.FloatTensor(state.flatten()).to(self.device), torch.FloatTensor([desire, elo]).to(self.device))
    
    def steps(self, states, desire, elo):
        s = [torch.FloatTensor(s.flatten()).to(self.device) for s in states]
        out_pos = self.forward(torch.stack(s), torch.FloatTensor([desire, elo]).to(self.device))
        out_neg = self.forward(torch.stack(s), torch.FloatTensor([-desire, elo]).to(self.device))

        return torch.softmax(out_pos - 0.1*out_neg, dim=1).cpu().detach().numpy()
        
    
    def forward(self, state, command):
        out = torch.relu(self.fc1(state))
        command_out = torch.relu(self.commands(command))
        out_first = out * command_out
        outx = out * command_out
        for i in range(0, self.layers):
            outx = torch.relu(self.fc[i](outx))
        out = self.last(outx)
        
        return out
    
    def action(self, state, desire, elo):
        """
        Samples the action based on their probability
        """
        action_prob = self.forward(state.expand(1, -1), desire, elo)[0,:]
        action_prob_2 = self.forward(state.expand(1, -1), desire*-1, elo)[0,:]
        probs = torch.softmax(action_prob-action_prob_2, dim=-1)
        action = torch.distributions.categorical.Categorical(probs=probs).sample()
        return probs.cpu().detach().numpy()
    
    def save(self, path):
        torch.save(self, "data/" + path + ".pth")
    def load(path):
        model = torch.load("data/" + path + ".pth")
        model.eval()
        return model