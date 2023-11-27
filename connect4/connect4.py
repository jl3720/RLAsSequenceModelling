import numpy as np
import enum
import copy


class strategy(enum.Enum):
    RANDOM = 1,
    MINMAX = 2

class connect4:
    def __init__(self) -> None:
        self.nrows = 6
        self.ncols =7 
        self.board = np.zeros((self.nrows,self.ncols))
        self.player_turn = 1
        self.winning_player = 0
        self.game_over = False

    def place_mark(self, col, player_id):
        current_col = self.board[:,col]
        spot_to_place = np.where(current_col == 0)[0]
        if len(spot_to_place)>0:
            self.board[spot_to_place[-1], col] = player_id
            self.check_game_over(player_id)
        else:
            throw: ValueError("Chosen column full")
        self.player_turn = self.player_turn % 2 + 1
    
    def check_game_over_other_board(self, board):
        self.board = board
        self.check_game_over()

    def check_game_over(self, player_id):
        
        # Check rows & cols
        winning_array = player_id*np.ones(4)
        for row in range(self.nrows):
            for col in range(self.ncols - 4 + 1):
                current_row = self.board[row,col:col +4 ]
                if np.array_equal(winning_array, current_row):
                    self.game_over = True
                    self.winning_player = player_id
                    
                    return
                
        for col in range(self.ncols):
            for row in range(self.nrows -4 + 1 ):
                current_col =self.board[row:row+4,col]

                if np.array_equal(winning_array, current_col.T):
                    self.game_over = True
                    self.winning_player = player_id
                    return


        # Check main diagonals
        for i in range(self.nrows - 4 + 1):
            for j in range(self.ncols - 4 + 1):
                if np.array_equal(np.diagonal(self.board[i:i+4, j:j+4]), winning_array):
                    self.game_over = True
                    self.winning_player = player_id

                    return 

        # Check anti-diagonals
        for i in range(4 - 1, self.nrows):
            for j in range(self.ncols - 4 + 1):
                if np.array_equal(np.diagonal(np.fliplr(self.board)[i-4+1:i+1, j:j+4]), winning_array):
                    self.game_over = True
                    self.winning_player = player_id
                    return 

    
    def get_board_state(self):
        return copy.copy(self.board)
    

class Player:
    def __init__(self, pid, strat, game) -> None:
        self.player_id = pid
        self.strategy = strat
        self.max_depth = 6
        self.game = game

    def place_mark(self):
        chosen_col = 0


        if self.strategy == strategy.RANDOM:
            chosen_col = self.get_random_col()

        else: 
            chosen_col = self.get_minmax_col(current_board, current_depth=0)
        
        self.game.place_mark(chosen_col, self.player_id)
    
    def get_minmax_col(self, current_board, current_depth):
        chosen_col = 0
        max_score = 0
        min_score = 0

        if current_depth == self.max_depth:
            possible_next_moves = self.get_possible_moves(self.game)
            for j in possible_next_moves:
                if j<0:
                    continue

                if connect4.check_game_over_other_board(current_board):

                    return 1
        self.get_minmax_col()


        return chosen_col
    def get_possible_moves(self):

        zero_positions = np.argmin(self.game.board == 0, axis = 0)
        full_columns = np.all(self.game.board != 0, axis = 0)
        zero_positions[full_columns] = -1
        zero_positions[zero_positions>0] -=1 
        return zero_positions

    def get_random_col(self):
        chosen_col = 0
        chosen = False

        while not chosen:
            suggestion = np.random.randint(self.game.ncols)
            if 0 in self.game.board[:,suggestion]:
                chosen = True
                chosen_col = suggestion

        return chosen_col

def main():
    game = connect4()
    p1 = Player(1, strategy.RANDOM, game)
    p2 = Player(2, strategy.RANDOM, game)
    turns_taken = 0
    for turn_counter in range(int(game.ncols*game.nrows / 2)):
        if not game.game_over:
            p1.place_mark()
        if not game.game_over:
            p2.place_mark()
        if game.game_over:
            turns_taken = turn_counter +1
            break

    print(game.get_board_state())
        
    if game.winning_player != 0:
        print(f"Game over, Player {game.winning_player} has won! They won in {turns_taken} turns.")
    else:
        print(f"Game ended in a draw.")



def test():
    pass
       

if __name__ == "__main__":
    main()
    