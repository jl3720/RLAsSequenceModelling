import numpy as np
import enum
import copy


class GameManager:
    p1_win = 0
    p2_win = 0
    draw = 0
    games = 0

    def __init__(self, players):
        self.players = players

    def info(self):
        print("p1: ", self.p1_win/self.games, "p2: ", self.p2_win/self.games, "draw: ", self.draw/self.games)

    # def _play_single(self, env, buffer, start=0):
    #     done = False
    #     state = env.reset()
    #     i = start
    #     while not done:
    #         state, action, done, rew = env.step(self.players[i%len(self.players)].step(state, 1))
    #         i += 1
    #     buffer.add_sample(*env.export_game())
    #     if start == 1:
    #         return -rew
    #     return rew
    
    def _play_multi(self, envs, buffer, elo_aim, start=0):
        rewards = [0 for env in envs]
        dones = [False for env in envs]
        states = [env.reset() for env in envs]
        
        # sample actions
        s_count = start
        while not np.all(dones):
            actions = self.players[s_count%2].steps(states, 1, elo_aim)
            for i, e in enumerate(envs):
                state, action, done, rew = envs[i].step(actions[i])
                if done and not dones[i]:
                    dones[i] = done
                    rewards[i] = rew
                states[i] = state
            s_count += 1
        
        if start == 1:
            [buffer.add_sample(*env.export_game(), [s.elo for s in [self.players[1], self.players[0]]]) for env in envs]
            return list(-1*np.array(rewards))
        else:
            [buffer.add_sample(*env.export_game(), [s.elo for s in self.players]) for env in envs]
        return rewards
    
    # fill the buffer with batch_size batches
    def play(self, batch_size, env_gen, buffer, elo_aim=1000, rated=True):
        #print("ELO Before: ", self.players[0].elo, self.players[1].elo)
        e_p1 = 1 / (1 + 10**((self.players[1].elo - self.players[0].elo)/400))
        e_p2 = 1 / (1 + 10**((self.players[0].elo - self.players[1].elo)/400))
        # play games in parallel really important

        rew = [*self._play_multi([env_gen() for _ in range(int(batch_size/2))], buffer, elo_aim, 0),
            *self._play_multi([env_gen() for _ in range(int(batch_size/2))], buffer, elo_aim, 1)]
        points_p1 = sum([0 if x == -1 else 1 if x == 1 else 0.5 for x in rew])
        points_p2 = sum([1 if x == -1 else 0 if x == 1 else 0.5 for x in rew])

        self.players[0].elo += np.round((batch_size/100) * 10*(points_p1/batch_size - e_p1))
        self.players[1].elo += np.round((batch_size/100) * 10*(points_p2/batch_size - e_p2))

        self.games += batch_size
        self.draw += rew.count(0)
        self.p1_win += rew.count(1)
        self.p2_win += rew.count(-1)

        #print("ELO After: ", self.players[0].elo, self.players[1].elo)
        return self.p1_win / self.games
        
        

        


class Player:
    elo = 1000 # default elo

    def step(self, state, desire, elo):
        pass

    # default implementation, different for pytorch model
    def steps(self, states, desire,elo):
        return [self.step(s, desire, elo) for s in states]

class RandomPlayer(Player):
    elo = 940
    def step(self, state, desire, elo_aim):
        return [1/7, 1/7, 1/7, 1/7, 1/7, 1/7, 1/7]
    
class GreedyRandomPlayer(Player):
    elo = 1060
    def step(self, state, desire, elo_aim):
        return [1/12, 1/12, 1/12, 0.5, 1/12, 1/12, 1/12]

class Connect4:
    def __init__(self) -> None:
        self.nrows = 6
        self.ncols =7

        self.player_turn = 1
        self.winning_player = 0
        self.game_over = False
        self.board = np.zeros((self.nrows,self.ncols))
        self.states = []
        self.actions = []

    def reset(self):
        self.player_turn = 1
        self.winning_player = 0
        self.game_over = False
        self.board = np.zeros((self.nrows,self.ncols))
        self.states = []
        self.actions = []
        return self.board.copy()

    # returns state, action, done, info
    # if done = True -> info 
    def step(self, action_prob):
        old_state = self.board.copy()
        if self.game_over:
            return old_state, 0, True, self.winning_player

        # first sample action_prob
        a = np.random.choice(range(self.ncols), 1, p=action_prob)[0]
        if not self.place_mark(a):
            x = sorted(range(self.ncols), key=lambda k: -action_prob[k])
            placed = False
            for act in x:
                if self.place_mark(act):
                    a = act
                    placed = True
                    break
            if not placed:
                return self.board.copy(), a, True, 0


        # secondly go throw all options 1 by 1
        self.check_game_over(self.player_turn)        
        self.player_turn = self.player_turn % 2 + 1

        state = self.board.copy()
        if self.player_turn != 1:
            state = -1*state

        self.states.append(old_state)
        self.actions.append(a)

        if self.game_over:            
            return state, a, True, self.winning_player
        return state, a, False, 0

    def export_game(self):
        return self.states, self.actions, self.winning_player

    def place_mark(self, col):
        current_col = self.board[:,col]
        spot_to_place = np.where(current_col == 0)[0]
        if len(spot_to_place)>0:
            self.board[spot_to_place[-1], col] = 1 if self.player_turn == 1 else -1
            return True
        return False

    def check_game_over(self, player_id):
        w = player_id
        if player_id == 2:
            w = -1

        # Check rows & cols
        winning_array = w*np.ones(4)
        for row in range(self.nrows):
            for col in range(self.ncols - 4 + 1):
                current_row = self.board[row,col:col +4 ]
                if np.array_equal(winning_array, current_row):
                    self.game_over = True
                    self.winning_player = w
                    
                    return
                
        for col in range(self.ncols):
            for row in range(self.nrows -4 + 1 ):
                current_col =self.board[row:row+4,col]

                if np.array_equal(winning_array, current_col.T):
                    self.game_over = True
                    self.winning_player = w
                    return


        # Check main diagonals
        for i in range(self.nrows - 4 + 1):
            for j in range(self.ncols - 4 + 1):
                if np.array_equal(np.diagonal(self.board[i:i+4, j:j+4]), winning_array):
                    self.game_over = True
                    self.winning_player = w

                    return 

        # Check anti-diagonals
        for i in range(4 - 1, self.nrows):
            for j in range(self.ncols - 4 + 1):
                if np.array_equal(np.diagonal(np.fliplr(self.board)[i-4+1:i+1, j:j+4]), winning_array):
                    self.game_over = True
                    self.winning_player = w
                    return 

    
    def get_board_state(self):
        return copy.copy(self.board)
    
