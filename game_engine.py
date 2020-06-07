import enum

import numpy as np


# Player types. Used both for training and playing the game.
class PlayerType(enum.Enum):
    HUMAN = "HUMAN"
    RANDOM = "RANDOM"
    DEEP_RL = "DEEP_RL"


# Game manager, for playing the game.
# Actively queries for each move, based on the player type.
class Manager:
    def __init__(self, player_x_type, player_o_type, random_order=False):
        assert player_x_type in PlayerType, "Invalid type {} for player x.".format(player_x_type)
        assert player_o_type in PlayerType, "Invalid type {} for player o.".format(player_x_type)

        # Map each player type to the method used to gather and make moves for that player type.
        executor_map = {
            PlayerType.HUMAN: self._make_human_move,
            PlayerType.RANDOM: self._make_random_move,
            PlayerType.DEEP_RL: self._make_deep_rl_move
        }

        # Executor method for each player.
        self._executors = {
            _Player.X: executor_map[player_x_type],
            _Player.O: executor_map[player_o_type]
        }
        if random_order and (np.random.rand() < 0.5):
            # If order is random, 50% chance to be O.
            self._current_player = _Player.O
        else:
            # All other cases (50% chance when random, default when not random) start player is X.
            self._current_player = _Player.X
        self._winner = None
        self._board = Board()
        self._moves_remaining = self._board.moves_remaining
        self._run_game()

    def _run_game(self):
        board = self._board
        while (self._winner is None) and (self._moves_remaining > 0):
            current_player = self._current_player
            is_winner = self._executors[current_player]()
            self._moves_remaining = board.moves_remaining
            if is_winner:
                self._winner = current_player
            self._switch_player()
        if self._winner is None:
            print("No winner. No more moves remaining.")
        else:
            print("Winner is {}!".format(self._winner))
        print("Final board:\n{}".format(board.get_grid()))

    def _switch_player(self):
        next_player = _Player(self._current_player.value * -1)
        self._current_player = next_player

    # Methods to gather and make moves for various player types.
    def _make_human_move(self):
        board = self._board
        current_player = self._current_player

        position = self._get_human_move()
        is_valid = board.validate_move(position)
        while not is_valid:
            position = self._get_human_move()
            is_valid = board.validate_move(position)
        is_winner = board.make_move(current_player, position)
        return is_winner

    def _get_human_move(self):
        grid = self._board.get_grid()
        print("Current board:\n{}".format(grid))
        if self._current_player == _Player.X:
            player_name = "X (+1)"
        else:
            player_name = "O (-1)"
        row = int(input("Row for player {}? (0 - 2)".format(player_name)))
        column = int(input("Column for player {}? (0 - 2)".format(player_name)))
        position = (row, column)
        return position

    def _make_random_move(self):
        is_winner = self._board.make_random_move(self._current_player)
        return is_winner

    def _make_deep_rl_move(self):
        pass


# Internal enum to track the players. Use +/-1 to represent them on the board.
class _Player(enum.Enum):
    X = 1
    O = -1


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
        assert player in _Player, "player {} is not valid.".format(player)
        position = self._validate_position(position)

        # Validate move.
        valid = self.validate_move(position)
        if not valid:
            raise ValueError("Specified position {} is taken. Move is not valid.".format(position))

        # Execute move.
        self._grid[position] = player.value
        self.last_player = player
        self.last_move = position
        self.moves_remaining -= 1

        # Check if the move produced a winner.
        is_winner = self.check_if_winner()
        return is_winner

    # Make a valid, random move.
    def make_random_move(self, player):
        # Array mask to knockout invalid positions.
        mask = 1 - np.abs(self._grid)
        # Generate a random value for each position.
        # Take the argmax after multiplying each element by the knockout mask.
        random_grid = np.random.rand(3, 3)
        masked_grid = np.multiply(random_grid, mask)
        # Have to regenerate the 2D index, since np.argmax flattens the array.
        random_position = np.unravel_index(np.argmax(masked_grid), masked_grid.shape)
        is_winner = self.make_move(player, random_position)
        return is_winner

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
