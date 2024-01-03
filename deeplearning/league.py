from connect4.connect4 import RandomPlayer, GreedyRandomPlayer, GameManager, Connect4
from deeplearning.buffer import ReplayBuffer
from deeplearning.mlp import BF
import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
import deeplearning.mlp as mlp
import torch.nn.functional as F

# this class contains all the logic related to training and testing and creating the agents
# start will be with 2 players (randomAgent and greedyRandomAgent)
# they will play 50.000 games afterwards a agent will be trained
# then from here on out the newest (trained!) agent will always self play 25.000 games and the rest of the 5.000 games will be split
# among the rest of the agents completly randomly

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class League():
    players: int = []
    season: int = 0
    bufferSize:int = 100000
    elo_history = {}

    def __init__(self, path="", season=0):
        self.season = season
        self.players = []

        self.players.append(RandomPlayer())
        self.players.append(GreedyRandomPlayer())

        self.buffer = ReplayBuffer(self.bufferSize*5)
        if len(path) > 0:
            print("Populating Buffer from file " + path)
            self.buffer = ReplayBuffer.load(path)

    def get_aim_elo(self):
        # Take the elo of the second highest rated player
        a = [s.elo for s in self.players]
        a.sort()
        return a[-2]

    def get_random_player(self):
        return self.players[np.random.randint(len(self.players))]
    
    def get_random_player_id(self):
        return np.random.randint(len(self.players))


    def play_season(self):
        # 0. If first season we need training data
        if self.season == 0:
            self.populate_first()

        # 1. Train new Agent
        agent = self.train_new_agent()
        self.players.append(agent)

        # 2. Elo Random Play
        print("Random Play")
        for _ in range(0, int(self.bufferSize/2/1000)):
            i1, i2 = 0,0
            while i1 == i2:
                i1, i2 = self.get_random_player_id(), self.get_random_player_id()
            print(str(i1), " vs ", str(i2))
            gm = GameManager([self.players[i1], self.players[i2]])
            gm.play(1000, Connect4, self.buffer, self.get_aim_elo())
            gm.info()

        # 3. Agent Self Play
        print("Agent Self Play")
        for _ in range(0, int(self.bufferSize/2/10000)):
            gm = GameManager([agent, agent])
            gm.play(10000, Connect4, self.buffer, self.get_aim_elo())

            

        # 4. Save Buffer + Save Agent
        print("Saving Agent")
        agent.save(str(self.season))
        print("Saving Buffer")
        self.buffer.save(str(self.season))

        print("Season " + str(self.season) + " results:")
        for i, a in enumerate(self.players):
            if not i in self.elo_history:
                self.elo_history[i] = [0 for _ in range(0, self.season)]
            self.elo_history[i].append(a.elo)
            print(i, a.elo)

        self.season += 1


    def populate_first(self):
        for _ in range(0, int(self.bufferSize/10000)):
            gm = GameManager([self.get_random_player(), self.get_random_player()])
            gm.play(10000, Connect4, self.buffer, self.get_aim_elo())

    def train_new_agent(self):
        print("Season " + str(self.season) + " training new Agent")

        bf = BF(42, 7, 5000, 1, device).to(device)
        optimizer = optim.Adam(params=bf.parameters(), lr=1e-5)

        i = 0
        cum_loss = 0
        best_loss = 100
        best_loss_i = 0
        while True:
            i += 1
            loss = train_behavior_function(1000, bf, optimizer, self.buffer)
            if loss < best_loss - 0.001:
                best_loss = loss
                best_loss_i = i
            if i - best_loss_i > 200:
                return bf
            cum_loss += loss
            if i % 100 == 0:
                print(i, cum_loss, best_loss_i, best_loss)
                cum_loss = 0


def train_behavior_function(batch_size, model, optimizer, buffer):
    """
    Trains the BF with on a cross entropy loss were the inputs are the action probabilities based on the state and command.
    The targets are the actions appropriate to the states from the replay buffer.
    """
    X, y = buffer.create_training_examples(batch_size)


    X = torch.stack(X)


    state = X[:,0:42]
    d = X[:,42:42+1]
    e = X[:,43:43+1]
    command = torch.cat([d,e], dim=-1)
    command2 = torch.cat([-d,e], dim=-1)
    y = torch.FloatTensor((y)).to(device).long()
    y_ = model(state.to(device), command.to(device)).float()
    optimizer.zero_grad()
    pred_loss = F.cross_entropy(y_, y)   
    pred_loss.backward()
    optimizer.step()
    return pred_loss.detach().cpu().numpy()