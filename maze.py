import random
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
import os
import time

from tensorflow.keras import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop, Adam, SGD
from keras.layers.advanced_activations import LeakyReLU
import random


class Game:
    up = 0
    left = 1
    down = 2
    right = 3

    actions = [0,1,2,3]
    names = ["UP", "LEFT", "DOWN", "RIGHT"]

    mv = {up: (1, 0), right: (0, 1), left: (0, -1), down: (-1, 0)}

    num_actions = len(actions)

    def __init__(self, n, m, n_hole, n_block, alea=False):
        self.n = n
        self.m = m
        self.alea = alea
        self.generate_game(n_hole, n_block)


    def _position_to_id(self, x, y):
        return x + y * self.n


    def _id_to_position(self, id):
        return (id % self.n, id // self.n)


    def generate_game(self,n_hole,n_block):
        cases = [(x, y) for x in range(self.n) for y in range(self.m)]

        holes_list = []
        for i in range(n_hole):
            hole=random.choice(cases)
            holes_list.append(hole)
            cases.remove(hole)

        block_list = []
        for j in range(n_block):
            block = random.choice(cases)
            block_list.append(block)
            cases.remove(block)

        start = random.choice(cases)
        cases.remove(start)
        end = random.choice(cases)
        cases.remove(end)

        self.position = start
        self.end = end
        self.hole = holes_list
        self.block = block_list
        self.counter = 0

        if not self.alea:
            self.start = start
        return self._get_state()


    def reset(self):
        if not self.alea:
            self.position = self.start
            self.counter = 0
            return self._get_state()
        else:
            return self.generate_game()

    def _get_grille(self, x, y):
        grille = [
            [0] * self.n for i in range(self.m)
        ]
        grille[x][y] = 1
        return grille

    def _get_state(self,rand=True):
        if not rand:
            if self.alea:
                return [self._get_grille(x, y) for (x, y) in
                        [self.position, self.end, self.hole, self.block]]
            return self._position_to_id(*self.position)
        else:
            if self.alea:
                return np.reshape([self._get_grille(x, y) for (x, y) in
                        [self.position, self.end, self.hole, self.block]],(1,4*self.n*self.m))
            return self._position_to_id(*self.position)



    def move(self, action):
        self.counter += 1

        if action not in self.actions:
            raise Exception("Invalid action")

        d_x, d_y = self.mv[action]
        x, y = self.position
        new_x, new_y = x + d_x, y + d_y

        if self.block.__contains__((new_x, new_y)):
            return self._get_state(), -1, False, self.actions
        elif self.hole.__contains__((new_x, new_y)):
            self.position = new_x, new_y
            return self._get_state(), -10, True, None
        elif self.end == (new_x, new_y):
            self.position = new_x, new_y
            return self._get_state(), 10, True, self.actions
        elif new_x >= self.n or new_y >= self.m or new_x < 0 or new_y < 0:
            return self._get_state(), -1, False, self.actions
        elif self.counter > 190:
            self.position = new_x, new_y
            return self._get_state(), -10, True, self.actions
        else:
            self.position = new_x, new_y
            return self._get_state(), -1, False, self.actions

    def print(self):
        str = ""
        for i in range(self.n - 1, -1, -1):
            for j in range(self.m):
                if (i, j) == self.position:
                    str += "x"
                elif self.block.__contains__((i,j)):
                    str += "¤"
                elif self.hole.__contains__((i,j)):
                    str += "o"
                elif (i, j) == self.end:
                    str += "@"
                else:
                    str += "."
            str += "\n"
        print(str)


def run(lr=0.85, size=[4,4],n_ep=200,n_hole=4,n_block=4,prob_disobediance=0., alea=False):
    state_n = size[0]*size[1]
    action_n = 4
    Q =np.zeros([state_n,action_n])

    y = 0.99
    total_reward_list = []
    actions_list = []
    states_list = []
    game = Game(size[0],size[1],n_hole=n_hole,n_block=n_block,alea=alea)

    for i in range(n_ep):
        action = []
        s = game.reset()
        state = [s]
        total_reward = 0
        d = False
        while True:
            if random.random() > prob_disobediance:
                Q2 = Q[s,:] + np.random.randn(1,action_n)*(1./(i+1))
            else:
                Q2 = Q[s,:] + np.random.randn(1,action_n)
            a = np.argmax(Q2)
            s1, reward, d, _ = game.move(a)
            Q[s, a] = Q[s, a] + lr * (reward + y * np.max(Q[s1, :]) - Q[s, a])  # Fonction de mise à jour de la Q-table

            total_reward += reward
            s = s1
            action.append(a)
            state.append(s)
            if d == True:
                break
        states_list.append(state)
        actions_list.append(action)
        total_reward_list.append(total_reward)

    game.reset()
    game.print()
    print("Score over time: " +  str(sum(total_reward_list[-100:])/100.0))

    plt.plot(total_reward_list)
    plt.ylabel('Cumulative reward')
    plt.xlabel('Étape')
    plt.show()


## CLASS TRAINER NEURAL NETWORK
class Trainer:
    def __init__(self, size=[4,4],learning_rate=0.01, epsilon_decay=0.9999):
        self.state_size = size[0]*size[1]
        self.action_size = 4
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.prepare()

    def prepare(self):
        model = Sequential()
        model.add(Dense(24, activation='relu'))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=SGD(lr=self.learning_rate))
        self.model = model

    def train(self, state, action, reward, next_state, done):
        target = self.model.predict(np.array([state]))[0]
        if done:
            target[action] = reward
        else:
            target[action] = reward + self.gamma * np.max(self.model.predict(np.array([next_state])))

        inputs = np.array([state])
        outputs = np.array([target])

        return self.model.fit(inputs, outputs, epochs=1, verbose=0, batch_size=1)

    def get_best_action(self, state, rand=True):

        self.epsilon *= self.epsilon_decay

        if rand and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        act_values = self.model.predict(np.array([state]))

        action = np.argmax(act_values[0])
        return action



def training(episodes, trainer, game):
    scores = []
    losses = [0]
    for e in range(episodes):
        state = game.reset()
        score = 0
        done = False
        steps = 0
        while not done:
            steps += 1
            action = trainer.get_best_action(state)
            next_state, reward, done, _ = game.move(action)
            score += reward
            trainer.train(state, action, reward, next_state, done)
            state = next_state
            if done:
                scores.append(score)
                break
            if steps > 200:
                trainer.train(state, action, -10, state, True) # we end the game
                scores.append(score)
                break
        if e % 10 == 0
            print("episode: {}/{}, moves: {}, score: {}"
                  .format(e, episodes, steps, score))
            print(f"epsilon : {trainer.epsilon}")
    return scores


def run2(lr=0.01, size=[4,4], n_ep=1000, n_hole=1, n_block=1, prob_disobediance=0.1, alea=False):
    game = Game(size[0],size[1],n_hole=n_hole,n_block=n_block,alea=alea) # Un jeu statique, avec 10% d'aléatoire dans les mouvements
    game.print()
    trainer = Trainer(learning_rate=lr)
    score = training(n_ep, trainer, game)

# run(lr=0.85, size=[6,6],n_ep=300,n_hole=5,n_block=5,prob_disobediance=0.05,alea=False)
run2(lr=0.01, size=[5,5], n_ep=300, n_hole=3, n_block=3, prob_disobediance=0.1, alea=False)


## Cas Aleatoire
class Trainer_map_alea:
    def __init__(self, size=[4,4],learning_rate=0.01, epsilon_decay=0.9999, batch_size=30, memory_size=3000):
        self.state_size = 4*size[0]*size[1]
        self.action_size = 4
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size

        model = Sequential()
        model.add(Dense(50, input_dim=self.state_size, activation='relu'))
        model.add(Dense(30, activation='relu'))
        model.add(Dense(30, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        self.model = model

        def remember(self, state, action, reward, next_state, done):
            self.memory.append([state, action, reward, next_state, done])

        def replay(self, batch_size):
            batch_size = min(batch_size, len(self.memory))

            minibatch = random.sample(self.memory, batch_size)

            inputs = np.zeros((batch_size, self.state_size))
            outputs = np.zeros((batch_size, self.action_size))

            for i, (state, action, reward, next_state, done) in enumerate(minibatch):
                target = self.model.predict(state)[0]
                if done:
                    target[action] = reward
                else:
                    target[action] = reward + self.gamma * np.max(self.model.predict(next_state))

                inputs[i] = state
                outputs[i] = target

        def decay_epsilon(self):
            self.epsilon *= self.epsilon_decay

        def get_best_action(self, state, rand=True):

            if rand and np.random.rand() <= self.epsilon:
                # The agent acts randomly
                return random.randrange(self.action_size)

            # Predict the reward value based on the given state
            act_values = self.model.predict(np.array(state))

            # Pick the action based on the predicted reward
            action = np.argmax(act_values[0])
            return action


def training2(episodes, trainer, wrong_action_p, alea, n_hole, n_block,size=[4,4], collecting=False, snapshot=5000):
    batch_size = 32
    g = Game(size[0],size[1], wrong_action_p, n_hole=n_hole, n_block=n_block, alea=alea)
    counter = 1
    scores = []
    global_counter = 0
    losses = [0]
    epsilons = []

    # we start with a sequence to collect information, without learning
    if collecting:
        collecting_steps = 10000
        print("Collecting game without learning")
        steps = 0
        while steps < collecting_steps:
            state = g.reset()
            done = False
            while not done:
                steps += 1
                action = g.get_random_action()
                next_state, reward, done, _ = g.move(action)
                trainer.remember(state, action, reward, next_state, done)
                state = next_state

    print("Starting training")
    global_counter = 0
    for e in range(episodes+1):
        state = g.generate_game()
        state = np.reshape(state, [1, 64])
        score = 0
        done = False
        steps = 0
        while not done:
            steps += 1
            global_counter += 1
            action = trainer.get_best_action(state)
            trainer.decay_epsilon()
            next_state, reward, done, _ = g.move(action)
            next_state = np.reshape(next_state, [1, 64])
            score += reward
            trainer.remember(state, action, reward, next_state, done)  # ici on enregistre le sample dans la mémoire
            state = next_state
            if global_counter % 100 == 0:
                l = trainer.replay(batch_size)   # ici on lance le 'replay', c'est un entrainement du réseau
                losses.append(l.history['loss'][0])
            if done:
                scores.append(score)
                epsilons.append(trainer.epsilon)
            if steps > 200:
                break
        if e % 200 == 0:
            print("episode: {}/{}, moves: {}, score: {}, epsilon: {}, loss: {}"
                  .format(e, episodes, steps, score, trainer.epsilon, losses[-1]))
        if e > 0 and e % snapshot == 0:
            trainer.save(id='iteration-%s' % e)
    return scores, losses, epsilons

def run3(lr=0.01, size=[4,4], n_ep=1000, n_hole=1, n_block=1, prob_disobediance=0.1, alea=False):
    trainer = Trainer_map_alea(learning_rate=lr, epsilon_decay=0.999995)
    scores, losses, epsilons = training2(n_ep, trainer, 0.1, True, size=size, snapshot=2500)
    print(scores)
    print(losses)
    print(epsilons)
