import enum

import numpy


# Game manager.
class Manager:
    def __init__(self):
        self._board = Board()
        self._current_player = Player.POSITIVE
        self._winner = None
        self._moves_remaining = self._board.moves_remaining
        self._run_game()

    def _run_game(self):
        board = self._board
        while (self._winner is None) and (self._moves_remaining > 0):
            position = self._get_move()
            is_valid = board.validate_move(position)
            while not is_valid:
                position = self._get_move()
                is_valid = board.validate_move(position)
            is_winner = board.make_move(self._current_player, position)
            self._moves_remaining = board.moves_remaining
            if is_winner:
                self._winner = self._current_player
            self._switch_player()
        if self._winner is None:
            print("No winner. No more moves remaining.")
        else:
            print("Winner is {}!".format(self._winner))

    def _get_move(self):
        grid = self._board.get_grid()
        print("Current board:\n{}".format(grid))
        if self._current_player == Player.POSITIVE:
            player_name = "One"
        else:
            player_name = "Two"
        row = int(input("Row for player {}? (0 - 2)".format(player_name)))
        column = int(input("Column for player {}? (0 - 2)".format(player_name)))
        position = (row, column)
        return position

    def _switch_player(self):
        next_player = Player(self._current_player.value * -1)
        self._current_player = next_player


# Use +/-1 for the two players.
class Player(enum.Enum):
    POSITIVE = 1
    NEGATIVE = -1


# The game board.
class Board:
    _down_diagonal_positions = {(0, 0), (1, 1), (2, 2)}
    _up_diagonal_positions = {(0, 2), (1, 1), (2, 0)}

    def __init__(self):
        self._grid = numpy.zeros([3, 3])
        self.last_player = None
        self.last_move = None
        self.moves_remaining = 9

    # Return a copy of the grid.
    def get_grid(self):
        return self._grid.copy()

    # Make a move and return whether this produces a winner.
    def make_move(self, player, position):
        # Validate input.
        assert player in Player, "player {} is not valid.".format(player)
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
