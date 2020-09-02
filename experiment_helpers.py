import math

import numpy as np
import pandas as pd

import game_engine as ge
import train


# Calculate percent of each game outcome, using a rolling window.
# Provides visibility into how the model performance evolves over the course of training.
def calculate_training_sequence(outcomes):
    outcomes = np.array(outcomes)
    num_games = len(outcomes)

    num_samples = min(1000, num_games)
    interval = math.ceil(num_games / num_samples)
    local_length = max(interval, min(int(num_games / 10), 100))

    rolling_percent_invalid = [np.vectorize(lambda x: int(x == train.Outcome.INVALID))(
        outcomes[n:(n + local_length)]).sum() / local_length
                              for n in range(0, len(outcomes) - local_length, interval)]
    rolling_percent_loss = [np.vectorize(lambda x: int(x == train.Outcome.LOSS))(
        outcomes[n:(n + local_length)]).sum() / local_length
                           for n in range(0, len(outcomes) - local_length, interval)]
    rolling_percent_tie = [np.vectorize(lambda x: int(x == train.Outcome.TIE))(
        outcomes[n:(n + local_length)]).sum() / local_length
                          for n in range(0, len(outcomes) - local_length, interval)]
    rolling_percent_win = [np.vectorize(lambda x: int(x == train.Outcome.WIN))(
        outcomes[n:(n + local_length)]).sum() / local_length
                          for n in range(0, len(outcomes) - local_length, interval)]
    return (rolling_percent_invalid, rolling_percent_loss, rolling_percent_tie, rolling_percent_win)


# Evaluate a model by playing against a random player or another model.
# Both player and opponent can either be a model, or 'random'.
def head_to_head(player, opponent='random', num_games=1000, verbose=True):
    if player == 'random':
        model_player = ge.RandomPlayer()
    else:
        model_player = ge.AIPlayer(player)
    if opponent == 'random':
        opponent_player = ge.RandomPlayer()
    else:
        opponent_player = ge.AIPlayer(opponent)

    outcomes = []
    # Run games and collect outcomes.
    for n in range(num_games):
        if verbose:
            if n % int(num_games/10) == 0:
                print("Game {} of {}.".format(n + 1, num_games))
        manager = ge.Manager(model_player, opponent_player, random_order=True, verbose=False)
        winner = manager.winner
        if winner is None:  # No winner, tie game.
            outcome = train.Outcome.TIE
        elif winner == ge.Symbol.X:  # Model won.
            outcome = train.Outcome.WIN
        elif winner == ge.Symbol.O:  # Model lost.
            outcome = train.Outcome.LOSS
        else:
            raise RuntimeError("Bug in code. Unexpected game winner.")
        outcomes.append(outcome)

    # Calculate summary stats.
    percent_win = np.vectorize(lambda x: int(x == train.Outcome.WIN))(outcomes).sum() / num_games
    percent_tie = np.vectorize(lambda x: int(x == train.Outcome.TIE))(outcomes).sum() / num_games
    percent_loss = np.vectorize(lambda x: int(x == train.Outcome.LOSS))(outcomes).sum() / num_games

    stats = {'percent_win': percent_win,
             'percent_tie': percent_tie,
             'percent_loss': percent_loss}
    return stats


# Find the move the model will make as first player in a new game.
# The center is the optimal move.
def evaluate_opening_move(model):
    ai = ge.AIPlayer(model)
    ai.symbol = ge.Symbol.X
    board = ge.Board()
    position = ai.make_move(board)
    return position


# Manage (save/load) metadata for trained models.
# Uses integer version numbers, auto-incrementing by one each time.
class Metadata:
    _instance = None
    # The file in which to save the metadata. Assumes local directory.
    _METADATA_FILE = 'metadata.csv'
    _VERSION_COLUMN = 'version'
    _METADATA_COLUMN = 'metadata'

    # Singleton class.
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            try:
                cls._instance._metadata = pd.read_csv(cls._METADATA_FILE, index_col=0)
                cls._instance._max_version = cls._instance._metadata[cls._VERSION_COLUMN].values[-1]
            except Exception as e:
                print(e)
                print("Unable to load metadata from file {}: {}. Initializing blank metadata.".format(cls._METADATA_FILE, type(e)))
                cls._instance._metadata = pd.DataFrame(columns=[cls._VERSION_COLUMN,
                                                                cls._METADATA_COLUMN])
                cls._instance._max_version = 0
        return cls._instance

    def add_experiment(self, metadata):
        next_version = self._max_version + 1
        self._metadata = self._metadata.append(pd.DataFrame({self._VERSION_COLUMN: [next_version],
                                                             self._METADATA_COLUMN: [metadata]}),
                                               ignore_index=True, sort=False)
        self._max_version = next_version
        return next_version

    def save(self):
        self._metadata.to_csv(self._METADATA_FILE)

    # Return a copy of the metadata DataFrame, with the metadata column exploded.
    def exploded_copy(self):
        def split_dict(row, key):
            result = pd.Series()
            for k, v in row[key].items():
                result[k] = v
            return result
        copy = self._metadata.apply(lambda x: split_dict(x, 'metadata'), axis=1)
        copy = self._metadata.join(copy)
        return copy
