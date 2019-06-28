import unittest
from utils import EnvFactory
from agent.mc.mc_control import MCControlAgent
from utils.hash_transformer import HashTransformer
from utils.dummy_transformer import DummyTransformer
from utils.plots import plot_V, plot_learning_curve
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns
import numpy as np
import pickle 
from copy import copy
from agent.policy import EpsGreedy, EpsDecayGreedy
import time
import os

RESULT_PATH = './results'

class Test(unittest.TestCase):
    def setUp(self):
        self.env = EnvFactory.getEnvironment('easy21-v0',0)
        np.random.seed(0)
        eps = EpsDecayGreedy(self.env.actions, 1000)
        self.agent = MCControlAgent(self.env, 1.0, HashTransformer(), eps, every_visit=True)

    def test1(self):
        player, dealer = self.env.observation_space.high
        print(player, dealer)
        episodes = 1000000
        start = time.time()    
        info = self.agent.learn(episodes)
        print(info)
        transf = self.agent.transformer
        print("Learning time: %s s" % (time.time() - start))
        V = self.agent.get_state_value_function(optimality=True)
        V_matrix = np.zeros((player, dealer))
        for k,v in V.items():
            state = transf.invtransform(k) - 1
            V_matrix[tuple(state.astype(int))] = v
        plot_V(V_matrix, dealer, player, save="mc1milionepisode.pdf")
        
        with open(os.path.join(RESULT_PATH,'VMC_{}.pickle'.format(episodes)), 'wb') as f:
             pickle.dump(V_matrix, f)

        Q_matrix = np.zeros((player, dealer, self.env.action_space.n))
        for k,v in self.agent.Q.items():
            state = tuple(self.agent.transformer.invtransform(k).astype(int) -1)
            Q_matrix[state] = v

        with open(os.path.join(RESULT_PATH,'QMC_{}.pickle'.format(episodes)), 'wb') as f:
            pickle.dump(Q_matrix, f)

        tests = 1000
        wl = [0, 0]
        Q = self.agent.Q
        for _ in range(tests):
            done = False
            observation = self.env.reset()
            reward = 0
            while not done:
                action = np.argmax(Q[self.agent.transformer.transform(observation)])
                observation, reward, done, _ = self.env.step(action)

            ix = 0 if reward == 1 else 1
            wl[ix] += 1
        print("Percentage of win: {}".format(wl[0]/tests))
            
