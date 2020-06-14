import math
import numpy as np
import pandas as pd
import tensorflow as tf

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
        outcomes[n:(n + local_length)]).sum() /
                              local_length
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


# Find the move the model will make as first player in a new game.
# The center is the optimal move.
def evaluate_opening_move(model):
    pass
