import unittest
from utils import EnvFactory
from agent.mc.mc_control import MCControlAgent, OffPolicyMCControlAgent, MCControlFA
from utils.hash_transformer import HashTransformer
from utils.dummy_transformer import DummyTransformer
from utils.plots import plot_V, plot_learning_curve
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns
import numpy as np
import pickle 
from copy import copy
from agent.policy import EpsGreedy, EpsDecayGreedy, Random
import time
from utils.features_transformer import LinearFeatureTransformer, PolynomialFeatureTransformer, TileFeatureTransformer
import estimators as e 
import os
import tests.usecases as uc

RESULT_PATH = './results'

class Test(unittest.TestCase):
    def setUp(self):
        self.env = EnvFactory.getEnvironment('easy21-v0',0)
        np.random.seed(0)
        
        
    @unittest.skip
    def test1(self):
        episodes = 100000
        eps = EpsDecayGreedy(self.env.actions, 1000)
        agent = MCControlAgent(self.env, 1.0, HashTransformer(), eps, every_visit=True)
        uc.learnQV(agent, episodes, 
                   self.env,
                   'mc',
                   save_Q=False,
                   save_V=False,
                   algo_kws=dict(episodes=episodes))

    @unittest.skip
    def test2(self):
        episodes = 100000
        eps = EpsDecayGreedy(self.env.actions, 1000)
        random = Random(self.env.actions)
        agent = OffPolicyMCControlAgent(self.env, 1.0, HashTransformer(), eps, random, every_visit=True)
        uc.learnQV(agent, episodes, 
                   self.env,
                   'offmc',
                   save_Q=False,
                   save_V=False,
                   algo_kws=dict(episodes=episodes))
    
    def test3(self):
        episodes = 10000
        eps = EpsDecayGreedy(self.env.actions, 1000)
        lin = LinearFeatureTransformer(self.env.observation_space.shape, self.env.action_space.n)
        state_shape, action_shape = lin.transformed_shape()
        es = e.LinearEstimator(state_shape[0]+action_shape)

        agent = MCControlFA(self.env, 1.0, HashTransformer(), eps,es, lin, 0.001, every_visit=True)
        agent.learn(episodes)
        player, dealer = self.env.observation_space.high
        V_matrix = np.zeros((player, dealer))
        for p in np.arange(0, player, 1):
            for d in np.arange(0, dealer):
                values = [es(lin.transform(np.array([p,d]), action=i)) for i in range(self.env.action_space.n)]
                V_matrix[int(p), int(d)] = max(values)
        plot_V(V_matrix, dealer, player)


