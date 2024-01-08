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
        self.fc1 = nn.Linear(state_space, hidden_size, bias=True)
        self.commands = nn.Linear(2, hidden_size, bias=True)
        self.layers = 5
        self.fc = [nn.Linear(hidden_size, hidden_size, bias=False).to(self.device) for x in range(0, self.layers)]
        self.last = nn.Linear(hidden_size, action_space, bias=False)

        self.sigmoid = nn.Sigmoid()

    def step(self, state, desire, elo):
        return self.steps([state], desire, elo)[0]
    
    def steps(self, states, desire, elo):
        s = [torch.FloatTensor(s.flatten()).to(self.device) for s in states]
        s_2 = [torch.FloatTensor(s.flatten()).to(self.device) for s in states]
        commands = [torch.FloatTensor([desire, elo]).to(self.device) for s in states]
        commands_2 = [torch.FloatTensor([-desire, elo]).to(self.device) for s in states]
        out_pos = self.forward(torch.stack(s), torch.stack(commands))
        out_neg = self.forward(torch.stack(s_2), torch.stack(commands_2))
        inner = out_pos - 0.5*out_neg

        return torch.softmax(inner, dim=1).cpu().detach().numpy()
        
    
    def forward(self, state, command):
        #print(state, command)
        out = torch.relu(self.fc1(state))
        command_out = torch.relu(self.commands(command))
        outx = out * command_out
        for i in range(0, self.layers):
            outx = torch.relu(self.fc[i](outx))
        out = self.last(outx)
        
        return out
    
    def save(self, path):
        torch.save(self, "data/" + path + ".pth")
    def load(path):
        model = torch.load("data/" + path + ".pth")
        model.eval()
        return model