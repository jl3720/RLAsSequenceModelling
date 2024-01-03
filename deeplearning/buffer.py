# for storing and loading past game rollouts
# as well as create training batches

import bz2
import pickle


import numpy as np
import torch


class ReplayBuffer():
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = []
        
        
    def add_sample(self, states, actions, reward, elo):
        episode = {"states": states, "actions":actions, "reward": reward, "elo": elo}
        self.buffer.append(episode)
    
    def sort(self):
        # keep the max buffer size
        self.buffer = self.buffer[:self.max_size]
    
    def get_random_samples(self, batch_size):
        self.sort()
        idxs = np.random.randint(0, len(self.buffer), batch_size)
        batch = [self.buffer[idx] for idx in idxs]
        return batch
    
    def save(self, file):
         with bz2.BZ2File("data/" + file + ".pbz2", "w") as f: 
            pickle.dump(self, f)

    def load(file):
        data = bz2.BZ2File("data/" + file + ".pbz2","rb")
        data = pickle.load(data)
        return data

    
    # FUNCTIONS FOR TRAINING
    def _select_time_steps(self, saved_episode):
        """
        Given a saved episode from the replay buffer this function samples random time steps (t1 and t2) in that episode:
        T = max time horizon in that episode
        Returns t1, t2 and T 
        """
        # Select times in the episode:
        T = len(saved_episode["states"]) # episode max horizon 
        t1 = int(np.random.power(1.5, 1)[0]*T)

        return t1

    def _create_training_input(self, episode, t):
        """
        Based on the selected episode and the given time steps this function returns 4 values:
        1. state at t1
        2. the desired reward: sum over all rewards from t1 to t2
        3. the time horizont: t2 -t1
        
        4. the target action taken at t1
        
        buffer episodes are build like [cumulative episode reward, states, actions, rewards]
        """

        fact = 1
        if t % 2 == 1:
            fact = -1

        state = fact*episode["states"][t]
        desired_reward = fact*episode['reward']
        action = episode["actions"][t]
        return state, desired_reward, action, episode['elo'][t%2]/1000.0

    def create_training_examples(self, batch_size):
        """
        Creates a data set of training examples that can be used to create a data loader for training.
        ============================================================
        1. for the given batch_size episode idx are randomly selected
        2. based on these episodes t1 and t2 are samples for each selected episode 
        3. for the selected episode and sampled t1 and t2 trainings values are gathered
        ______________________________________________________________
        Output are two numpy arrays in the length of batch size:
        Input Array for the Behavior function - consisting of (state, desired_reward, time_horizon)
        Output Array with the taken actions 
        """
        input_array = []
        output_array = []
        # select randomly episodes from the buffer
        episodes = self.get_random_samples(batch_size)
        for ep in episodes:
            #select time stamps
            t = self._select_time_steps(ep)

            state, desired_reward, action, elo = self._create_training_input(ep, t)
            input_array.append(torch.cat([torch.FloatTensor(state.flatten()), torch.FloatTensor([desired_reward, elo])]))
            output_array.append(action)
        return input_array, output_array
    
    def __len__(self):
        return len(self.buffer)