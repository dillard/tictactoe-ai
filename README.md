# tictactoe-ai
Python scripts to train an AI to play Tic-Tac-Toe using Q Reinforcement Learning.

## Introduction
Use `ModelTrainer.ipynb` or `train.q_learning()` to train a model.

Play a game with `GamePlay.ipynb` or `game_engine.Manager()`.

## Setup
The Python files were developed with:
- python 3.5.4
- numpy 1.18.5
- tensorflow 2.2.0

The `ModelTrainer.ipynb` notebook file additionally uses:
- matplotlib 3.0.3
- pandas 0.25.3

## Basic AI
Use the defaults in `ModelTrainer.ipynb` to train a basic model that performs moderately better 
than a random player (which makes moves randomly). The exact model will vary due to randomness in 
the training process. However, you should get a model that beats a random player ~55% of the time 
and loses ~40% (tying the remainder). This improves on the performance of a random player vs random 
player.

| Outcome vs Random Player | Basic AI (`example_model`) | Random Player |
| :---: | :---: | :---: |
| Win | 56% | 43% |
| Tie | 6% | 13% |
| Lose | 37% | 43% |

An example model is provided in the `example_model` directory. To load it to memory and play a game 
against the AI, use `GamePlay.ipynb`.

To improve the AI performance:
- train over more games
- adjust the Q-learning parameters
- use different deep learning architectures
