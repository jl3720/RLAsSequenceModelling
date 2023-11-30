import numpy as np
import enum
import copy


class GameManager:
    def __init__(self, players):
        self.players = players

    def _play_single(self, env, buffer):
        done = False
        state = env.reset()
        i = 0
        while not done:
            state, action, done, rew = env.step(self.players[i%len(self.players)].step(state, 1))
        buffer.add_sample(*env.export_game())

    # fill the buffer with batch_size batches
    def play(self, batch_size, env, buffer):
        for i in range(0, batch_size):
            self._play_single(env, buffer)


class Player:
    def step(self, state, desire):
        pass

class RandomPlayer(Player):
    def step(self, state, desire):
        return [1/7, 1/7, 1/7, 1/7, 1/7, 1/7, 1/7]

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
                return self.board.copy(), a, True, self.winning_player


        # secondly go throw all options 1 by 1
        self.check_game_over(self.player_turn)        
        self.player_turn = self.player_turn % 2 + 1

        state = self.board.copy()
        if self.player_turn != 1:
            state = -1*state

        self.states.append(state)
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
    
