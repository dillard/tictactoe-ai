import enum

import numpy as np

import game_engine as ge


# Outcomes of training games.
class Outcome(enum.Enum):
    INVALID = -1  # Invalid move
    LOSS = 0
    TIE = 1
    WIN = 2


# Training code, using Q reinforcement learning.
# The model should take an array of shape (1, 9) as input and produce an array of shape (1, 9) as
# output.
# rewards is a 4-tuple, specifying rewards for:
    # making an invalid move
    # losing the game
    # tying the game
    # winning the game
# Returns a tuple of (outcomes, metadata), where
    # outcomes is a list with the outcome for each game
    # metadata is a dict of training paramenters
# Code adapted from:
# https://adventuresinmachinelearning.com/reinforcement-learning-tutorial-python-keras/
# https://github.com/adventuresinML/adventures-in-ml-code/blob/master/r_learning_python.py
def q_learning(model, num_games, y=0.95, eps=0.5, decay_factor=0.999,
               rewards=(0, 0.25, 0.5, 2), verbose=True):
    # Capture hyperparameters used in training:
    metadata = {'num_games': num_games,
                'y': y,
                'eps': eps,
                'decay_factor': decay_factor,
                'rewards': rewards,
                'opponent': 'random',
                'state_values': (1, 0, -1)}

    outcomes = []
    for i in range(num_games):
        training_manager = ge.TrainingManager("random")
        state = training_manager.start_game()
        eps *= decay_factor
        if verbose:
            if i % int(num_games/10) == 0:
                print("Game {} of {}.".format(i + 1, num_games))
        game_finished = False
        while not game_finished:
            inference = model.predict(np.ravel(state).reshape((1, 9)))
            # Explore using a random action with decaying probability.
            if np.random.random() < eps:
                move = np.random.randint(0, 9)
            else:
                move = np.argmax(inference)
            position = np.unravel_index(move, (3, 3))
            new_state, move_valid, game_finished, is_winner = training_manager.make_move(position)
            # Determine reward (and, if the game is finished, the outcome).
            if not move_valid:
                # On an invalid move, stop the game and proceed to the next game.
                reward = rewards[0]
                outcome = Outcome.INVALID
                game_finished = True
            elif is_winner is True:
                assert game_finished, "Bug in code. Expect game to be finished."
                reward = rewards[3]
                outcome = Outcome.WIN
            elif is_winner is False:
                assert game_finished, "Bug in code. Expect game to be finished."
                reward = rewards[1]
                outcome = Outcome.LOSS
            else:
                reward = rewards[2]
                outcome = Outcome.TIE
            # Create target vector for training. Only modify the value for the move chosen.
            target = reward + y * np.max(model.predict(np.ravel(new_state).reshape((1, 9))))
            target_vec = inference
            target_vec[0][move] = target
            # Update the model.
            model.fit(np.ravel(state).reshape((1, 9)), target_vec, shuffle=False, verbose=0)
            state = new_state
        outcomes.append(outcome)
    return (outcomes, metadata)
