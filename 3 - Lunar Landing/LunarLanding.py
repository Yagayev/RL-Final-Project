from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam, SGD
from keras.models import load_model
import numpy as np
import gym
from collections import deque
import random
import sys
import csv

neural_network_name = "LunarNeuralNetwork"
all_scores = "allscores.csv"
averages = "averages.csv"

class LunarLandingNeuralNetwork:
    def __init__(self):
        self.alpha = 0.0001
        self.action_space = [0, 1, 2, 3]
        self.action_space_size = 4
        self.input_size = 11

        # building network
        self.model = Sequential()
        self.model.add(Dense(self.input_size, input_dim=self.input_size))  # input-observ
        self.model.add(Dense(300, activation='relu'))
        self.model.add(Dense(300, activation='relu'))
        self.model.add(Dense(self.action_space_size, activation='linear'))  # outpot-actions
        self.model.compile(loss='mse', optimizer=Adam(lr=self.alpha))

    # return array of the scores of each action
    def predict(self, state):
        state = np.reshape(state, [1, self.input_size])
        return self.model.predict(state)

    # return the best ACTION
    def find_best_action(self, state):
        actions_scores = self.predict(state)
        return np.argmax(actions_scores) # the INDEX of the maximum value in the array

    # return the SCORE of the best action
    def evaluate_state(self, state):
        actions_scores = self.predict(state)
        return np.max(actions_scores)  # the VALUE of the maximum value in the array



class LunarLandingNeuralAgent:

    def __init__(self):
        self.epsilon = 1
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.997
        self.gamma = 0.99
        self.batch_size = 50
        self.action_space = [0, 1, 2, 3]
        self.input_size = 11
        self.memory = deque(maxlen=500000)
        self.network = LunarLandingNeuralNetwork()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act_epsilon_greedy(self, state):
        if np.random.random() < self.epsilon:
            return self.act_random()
        else:
            return self.act_greedy(state)

    def act_random(self):
        return np.random.choice(self.action_space)

    def act_greedy(self, state):
        return self.network.find_best_action(state)

    def update(self):
        if len(self.memory) < self.batch_size:
            # not enough states to learn yet
            return


        minibatch = random.sample(self.memory, self.batch_size)
        batch_states = []
        batch_target_scores = []

        for state, action, reward, next_state, done in minibatch:
            batch_states.append(np.reshape(state, [1, self.input_size])[0])
            target_scores = self.network.predict(state)
            if not done:
                reward += self.gamma * self.network.evaluate_state(next_state)
            target_scores[0][action] = reward
            target = np.reshape(target_scores, [1, 4])
            # act = np.zeros(4)
            # act[action] = 1
            batch_target_scores.append(target[0])
        # print("LEARNING x:", np.array(batch_states).shape, '\n\ny:', np.array(batch_target_scores).shape)
        self.network.model.fit(np.array(batch_states), np.array(batch_target_scores), batch_size=len(minibatch), verbose=0)

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon * self.epsilon_decay


def train(agent, env, skip_size):
    scores = []
    score_table = []
    # state { pos-X, pos-Y, vel-X, vel-Y, angle, angular Vel, left leg touching, right leg touching}
    # Nop, fire left engine, main engine, right engine
    for i in range(skip_size):
        observation = env.reset()
        state = [i for i in observation]
        # appending "Accelerations" (no prerivious velocity):
        state.append(observation[2] - 0)
        state.append(observation[3] - 0)
        state.append(observation[5] - 0)
        done = False
        total_score = 0

        while not done:
            action = agent.act_epsilon_greedy(state)

            new_observation, reward, done, _ = env.step(action)
            new_state = [i for i in new_observation]
            # adding accelerations
            new_state.append(observation[2] - state[2])
            new_state.append(observation[3] - state[3])
            new_state.append(observation[5] - state[5])
            total_score += reward
            agent.remember(state, action, reward, new_state, done)
            state = new_state
            agent.update()
        # print("score = ", total_score)
        score_table.append([total_score])
        scores.append(total_score)
        agent.update_epsilon()

    return np.mean(scores), score_table


def play(agent, env):
    
    observation = env.reset()
    state = [i for i in observation]
    # appending "Accelerations" (no prerivious velocity):
    state.append(observation[2] - 0)
    state.append(observation[3] - 0)
    state.append(observation[5] - 0)
    done = False
    total_score = 0

    while not done:
        env.render()
        action = agent.act_epsilon_greedy(state)
        new_observation, reward, done, _ = env.step(action)
        new_state = [i for i in new_observation]
        # adding accelerations
        new_state.append(observation[2] - state[2])
        new_state.append(observation[3] - state[3])
        new_state.append(observation[5] - state[5])
        total_score += reward
        state = new_state
    print("score:", total_score)
    return


def main():
    env = gym.make('LunarLander-v2')
    agent = LunarLandingNeuralAgent()
    skip_size = 50
    total_games = 20000
    f = open(averages, 'a', newline='')
    with f:
        writer = csv.writer(f)
        writer.writerows([["episode batch",  "avg_scores",  "epsilon",  "alpha"]])
    for i in range(0, total_games, skip_size):
        avg_scores, score_table = train(agent, env, skip_size)
        print("[{}] avg: {}\tepsilon:{}\talpha:{}".format(i, avg_scores, agent.epsilon, agent.network.alpha))
        sys.stdout.flush()
        # play(agent)
        f = open(averages, 'a', newline='')
        with f:
            writer = csv.writer(f)
            writer.writerows([[i, avg_scores, agent.epsilon, agent.network.alpha]])
        f = open(all_scores, 'a', newline='')
        with f:
            writer = csv.writer(f)
            writer.writerows(score_table)
        if i % 100 == 0 or i == 50:

            agent.network.model.save('./models/'+neural_network_name+str(i)+'.h5')




def main2():
    agent = LunarLandingNeuralAgent()
    i = 9000
    agent.network.model = load_model('./models/' + neural_network_name + str(i) + '.h5')
    agent.epsilon = 0.01
    env = gym.make('LunarLander-v2')
    for i in range(20):
        play(agent, env)




if __name__ == "__main__":
    main()