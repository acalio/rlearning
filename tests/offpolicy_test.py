import unittest
from utils import EnvFactory
from agent.mc.mc_control import MCControlAgent, OffPolicyMCControlAgent
from utils.hash_transformer import HashTransformer
from utils.plots import plot_V, plot_learning_curve
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle 
from agent.policy import EpsGreedy,Random
import time
from copy import copy

class Test(unittest.TestCase):
    def setUp(self):
        self.env = EnvFactory.getEnvironment('easy21-v0',0)
        np.random.seed(0)
        eps = EpsGreedy(self.env.actions, 0.5)
        ran = Random(self.env.actions)
        self.agent = OffPolicyMCControlAgent(self.env, 1.0, HashTransformer(), eps, ran)

    def test1(self):
        player, dealer = self.env.observation_space.high
        print(player, dealer)
        episodes = 1000
        start = time.time()    
        info = self.agent.learn(episodes)
        print("Learning time: %s s" % (time.time() - start))
        transf = self.agent.transformer
        V = self.agent.get_state_value_function(optimality=True)
        V_matrix = np.zeros((player, dealer))
        for k,v in V.items():
            state = transf.invtransform(k) - 1
            V_matrix[tuple(state.astype(int))] = v
        #plot_V(V_matrix, dealer, player)
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
            
