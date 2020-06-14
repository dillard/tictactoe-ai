import abc
import enum

import numpy as np


# Abstract class for creating players.
class Player(abc.ABC):
    def __init__(self):
        self._symbol = None

    @property
    def symbol(self):
        return self._symbol

    # The game manager sets the symbol (X, O) for each player before any moves are made.
    @symbol.setter
    def symbol(self, symbol):
        assert symbol in Symbol, "Invalid symbol."
        self._symbol = symbol

    @abc.abstractmethod
    def make_move(self, board):
        pass


# Collects moves via prompt.
class HumanPlayer(Player):
    def make_move(self, board):
        grid = board.get_grid()
        position = self._get_move(grid)
        is_valid = board.validate_move(position)
        # If move is involid, keep prompting until receive a valid move.
        while not is_valid:
            print("Position {} is already filled or invalid. Make a different move.".format(position))
            position = self._get_move(grid)
            is_valid = board.validate_move(position)
        return position

    def _get_move(self, grid):
        if self.symbol is None:
            raise RuntimeError("Must set the symbol for this player.")
        print("Current board:\n{}".format(grid))
        player_name = self.symbol.name
        row = int(input("Row for player {}? (0 - 2)".format(player_name)))
        column = int(input("Column for player {}? (0 - 2)".format(player_name)))
        position = (row, column)
        return position


# Makes random moves.
class RandomPlayer(Player):
    def make_move(self, board):
        position = board.get_random_move()
        return position


# Uses a trained model to make moves.
class AIPlayer(Player):
    def __init__(self, model):
        super().__init__()
        self._model = model

    def make_move(self, board):
        state = board.get_grid()
        inference = self._model.predict(np.ravel(state).reshape((1, 9)))
        # Evaluate each potential move (ordered by inferred reward), taking the first valid one.
        moves = sorted(zip(inference[0], range(9)), key=lambda x: x[0], reverse=True)
        for (reward, m) in moves:
            position = np.unravel_index(m, (3, 3))
            is_valid = board.validate_move(position)
            if is_valid:
                break
        return position


# Game manager for training an RL model.
# Provides methods for the training function to call for game setup and making a move.
class TrainingManager:
    # The RL model under training is always player O.
    # player_x accepts either:
        # instance of Player
        # string: ("human", "random") to create a player of that type
    def __init__(self, player_x, random_order=True):
        assert isinstance(player_x, (Player, str)), "Invalid player_x."

        # Generate player X, if not already passed in.
        if not isinstance(player_x, Player):
            player_x = Manager.generate_player(player_x)
        player_x.symbol = Symbol.X
        self._player_x = player_x

        # Determine starting player
        if random_order and (np.random.rand() < 0.5):
            # If order is random, 50% chance to start with the training player (O).
            self._current_player = Symbol.O
        else:
            # All other cases (50% chance when random, default when not random) start with the
            # non-training player (X).
            self._current_player = Symbol.X

        self._winner = None
        self._board = Board()
        self._moves_remaining = self._board.moves_remaining

    # Return the starting game state, after making a move for player X, if needed.
    def start_game(self):
        if self._current_player == Symbol.X:
            board = self._board
            position = self._player_x.make_move(board)
            board.make_move(self._current_player, position)
            self._switch_player()
        return self._board.get_grid()

    # Method for the training player to make a move.
    # Resolves both the training player's move and the opponent's move (if any).
    # Returns a tuple of:
        # the game state (board grid): numpy array
        # whether the attempted move was valid: boolean
        # whether the game is over: boolean
        # whether the move resulted in the training player winning:
            # True: training player has won the game
            # False: training player has lost the game (opponent has won)
            # None: no winner
    def make_move(self, position):
        assert self._moves_remaining > 0, "No more moves remaining. The game is over."
        assert self._winner is None, "The game is over, with a winner."
        assert self._current_player == Symbol.O, "Bug in code. Expect Player O to be current player."
        board = self._board

        # Validate the move. If invalid, return an unchanged game state.
        is_valid = board.validate_move(position)
        if not is_valid:
            grid = board.get_grid()
            return (grid, is_valid, False, None)

        # Make the move and check if the game is over. If the game is over, return.
        is_winner = board.make_move(self._current_player, position)
        self._moves_remaining = board.moves_remaining
        if is_winner:
            self._winner = self._current_player
            grid = board.get_grid()
            return (grid, is_valid, True, True)
        if self._moves_remaining == 0:
            grid = board.get_grid()
            return (grid, is_valid, True, None)

        # Otherwise, the game continues with the other player (X). Again, return if the game has
        # ended.
        self._switch_player()
        if is_winner:
            self._winner = self._current_player
            position = self._player_x.make_move(board)
            board.make_move(self._current_player, position)
            grid = board.get_grid()
            return (grid, is_valid, True, False)
        if board.moves_remaining == 0:
            grid = board.get_grid()
            return (grid, is_valid, True, None)

        # The game hasn't ended. Return the game state.
        self._switch_player()
        grid = board.get_grid()
        return (grid, is_valid, False, None)

    def _switch_player(self):
        if self._current_player == Symbol.X:
            next_player = Symbol.O
        else:
            next_player = Symbol.X
        self._current_player = next_player


# Game manager, for playing the game.
# Actively queries for each move, based on the player type.
class Manager:
    # player_x and player_x accept either:
        # instance of Player
        # string: ("human", "random") to create a player of that type
    def __init__(self, player_x, player_o, random_order=False):
        assert isinstance(player_x, (Player, str)), "Invalid player_x."
        assert isinstance(player_o, (Player, str)), "Invalid player_x."

        # Generate Player instances, if not already passed.
        if not isinstance(player_x, Player):
            player_x = self.generate_player(player_x)
        if not isinstance(player_o, Player):
            player_o = self.generate_player(player_o)

        # Set symbol for each player.
        player_x.symbol = Symbol.X
        player_o.symbol = Symbol.O

        if random_order and (np.random.rand() < 0.5):
            # If order is random, 50% chance to start with player O.
            self._play_order = (player_o, player_x)
        else:
            # All other cases (50% chance when random, default when not random) start player is X.
            self._play_order = (player_x, player_o)
        self._winner = None
        self._board = Board()
        self._moves_remaining = self._board.moves_remaining
        self._run_game()

    @property
    def winner(self):
        return self._winner

    @property
    def moves_remaining(self):
        return self._moves_remaining

    @classmethod
    def generate_player(cls, player_type):
        if player_type == "human":
            player = HumanPlayer()
        elif player_type == "random":
            player = RandomPlayer()
        else:
            raise ValueError("Invalid player specification {}. Must be 'human' or 'random'.".format(player_type))
        return player

    def _run_game(self):
        board = self._board
        while (self._winner is None) and (self._moves_remaining > 0):
            current_player = self._play_order[0]
            position = current_player.make_move(board)
            is_winner = board.make_move(current_player.symbol, position)
            self._moves_remaining = board.moves_remaining
            if is_winner:
                self._winner = current_player.symbol
            self._switch_player()
        if self._winner is None:
            print("No winner. No more moves remaining.")
        else:
            print("Winner is player {}!".format(self._winner.name))
        print("Final board:\n{}".format(board.get_grid()))

    def _switch_player(self):
        self._play_order = self._play_order[::-1]


# Symbols used by each player. Use +/-1 to represent them on the board.
class Symbol(enum.Enum):
    X = (1, "X")
    O = (-1, "O")

    def __init__(self, board_value, name):
        self._board_value = board_value
        self._name = name

    @property
    def board_value(self):
        return self._board_value

    @property
    def name(self):
        return self._name


# The game board.
class Board:
    _down_diagonal_positions = {(0, 0), (1, 1), (2, 2)}
    _up_diagonal_positions = {(0, 2), (1, 1), (2, 0)}

    def __init__(self):
        self._grid = np.zeros((3, 3))
        self.last_player = None
        self.last_move = None
        self.moves_remaining = 9

    # Return a copy of the grid.
    def get_grid(self):
        return self._grid.copy()

    # Make a move and return whether this produces a winner.
    def make_move(self, player, position):
        # Validate input.
        assert player in Symbol, "player {} is not valid.".format(player)
        position = self._validate_position(position)

        # Validate move.
        valid = self.validate_move(position)
        if not valid:
            raise ValueError("Specified position {} is taken. Move is not valid.".format(position))

        # Execute move.
        self._grid[position] = player.board_value
        self.last_player = player
        self.last_move = position
        self.moves_remaining -= 1

        # Check if the move produced a winner.
        is_winner = self.check_if_winner()
        return is_winner

    # Generate a valid, random move.
    def get_random_move(self):
        # Array mask to knockout invalid positions.
        mask = 1 - np.abs(self._grid)
        # Generate a random board_value for each position.
        # Take the argmax after multiplying each element by the knockout mask.
        random_grid = np.random.rand(3, 3)
        masked_grid = np.multiply(random_grid, mask)
        # Have to regenerate the 2D index, since np.argmax flattens the array.
        random_position = np.unravel_index(np.argmax(masked_grid), masked_grid.shape)
        return random_position

    # Check whether a move is valid, without making it.
    def validate_move(self, position):
        valid = True
        try:
            position = self._validate_position(position)
        except AssertionError:
            valid = False
        except ValueError:
            valid = False
        else:
            if self._grid[position] != 0:
                valid = False
        return valid

    def _validate_position(self, position):
        assert len(position) == 2, "position must be length 2."
        for val in position:
            assert val >= 0, "Values in position must be >= 0."
            assert val <= 2, "Values in position must be <= 2."
        position = tuple(position)  # Used for indexing the numpy array.
        return position

    # Check if the last move produced a winner.
    def check_if_winner(self):
        grid = self._grid
        position = self.last_move
        is_winner = False
        # Check row.
        if abs(grid[position[0]].sum()) == 3:
            is_winner = True
        # Check column.
        if (is_winner is False) and abs(grid[:, position[1]].sum()) == 3:
            is_winner = True
        # Check diagonals.
        if (is_winner is False) and (position in self._down_diagonal_positions):
            total = sum([grid[p] for p in self._down_diagonal_positions])
            if abs(total) == 3:
                is_winner = True
        if (is_winner is False) and (position in self._up_diagonal_positions):
            total = sum([grid[p] for p in self._up_diagonal_positions])
            if abs(total) == 3:
                is_winner = True
        return is_winner
