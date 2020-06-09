import numpy as np
import keras
#from keras.models import Sequential
#from keras.layers import Dense, InputLayer

import game_engine as ge

# Code adapted from:
# https://adventuresinmachinelearning.com/reinforcement-learning-tutorial-python-keras/
# https://github.com/adventuresinML/adventures-in-ml-code/blob/master/r_learning_python.py


def q_learning_keras(env, num_episodes=1000):
    # create the keras model
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(batch_input_shape=(3, 3)))
    model.add(keras.layers.Dense(10, activation='sigmoid'))
    model.add(keras.layers.Dense(2, activation='linear'))
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    # now execute the q learning
    y = 0.95
    eps = 0.5
    decay_factor = 0.999
    r_avg_list = []
    for i in range(num_episodes):
        training_manager = ge.TrainingManager()
        state = training_manager.start_game()
        eps *= decay_factor
        if i % 100 == 0:
            print("Episode {} of {}".format(i + 1, num_episodes))
        done = False
        r_sum = 0
        while not done:
            if np.random.random() < eps:
                a = np.random.randint(0, 2)
            else:
                a = np.argmax(model.predict(np.identity(5)[state:state + 1]))
            new_s, r, done, _ = env.step(a)
            target = r + y * np.max(model.predict(np.identity(5)[new_s:new_s + 1]))
            target_vec = model.predict(np.identity(5)[state:state + 1])[0]
            target_vec[a] = target
            model.fit(np.identity(5)[state:state + 1], target_vec.reshape(-1, 2), epochs=1, verbose=0)
            state = new_s
            r_sum += r
        r_avg_list.append(r_sum / 1000)
    for i in range(5):
        print("State {} - action {}".format(i, model.predict(np.identity(5)[i:i + 1])))