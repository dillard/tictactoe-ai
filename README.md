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
the training process. However, you should get a model that beats a random player ~60% of the time 
and loses ~25% (tying the remainder). This improves on the performance of a random player vs random 
player.

| Outcome vs Random Player | Basic AI (`example_model_1`) | Random Player |
| :---: | :---: | :---: |
| Win | **63%** | 43% |
| Tie | 12% | 13% |
| Lose | **25%** | 43% |

An example model is provided in the `example_model_1` directory. To load it to memory and play a 
game against the AI, use `GamePlay.ipynb`.

## Improved AI
To improve the AI performance:
- train over more games
- adjust the Q-learning hyperparameters
- use different deep learning architectures

For example, increasing the number of training games by 10x (`example_model_2`), raises the win 
percentage to ~75% and decreases the loss percentage to ~15%. Tic-tac-toe is a solved game and, 
when played perfectly, a player can always avoid losing. So the decreased loss percentage 
demonstrates progress, but there's still room for improvement in this AI.

| Outcome vs Random Player | Basic AI (`example_model_1`) | Intermediate AI: 10x training (`example_model_2`) | Random Player |
| :---: | :---: | :---: | :---: |
| Win | 63% | **76%** | 43% |
| Tie | 12% | 8% | 13% |
| Lose | 25% | **16%** | 43% |
