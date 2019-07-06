import unittest
from utils import EnvFactory
from agent.tdlearning.td_control import SARSA, ExpSARSA, QLearning, DoubleQLearning
from utils.hash_transformer import HashTransformer
from utils.dummy_transformer import DummyTransformer
from utils.plots import plot_V, plot_learning_curve, plot_learning_curve_
import matplotlib.pyplot as plt
import tests.usecases as uc
from collections import defaultdict
import seaborn as sns
import numpy as np
import pickle 
from copy import copy
from agent.policy import EpsGreedy, EpsDecayGreedy, Random
import time
import os

RESULT_PATH = './results'



class Test(unittest.TestCase):
    def setUp(self):
        self.env = EnvFactory.getEnvironment('easy21-v0',0)
        self.episodes = 50000
        
    def test1(self):
        alpha = 0.01
        np.random.seed(0)
        eps = EpsDecayGreedy(self.env.actions, 100)
        agent = SARSA(self.env, 1.0, HashTransformer(), eps, alpha)
        uc.learnQV(agent, self.episodes,
                   self.env,
                   'SARSA',
                   save_Q=False,
                   save_V=False,
                   algo_kws=dict(episodes=self.episodes, alpha=alpha))     
        uc.play(agent, self.env, 1000)   

    def test2(self):
        alpha = 0.01
        np.random.seed(0)
        eps = EpsDecayGreedy(self.env.actions, 100)
        agent = ExpSARSA(self.env, 1.0, HashTransformer(), eps, alpha)
        uc.learnQV(agent,
                   self.episodes,
                   self.env, 'EXP_SARSA',
                   save_Q=False,
                   save_V=False,
                   algo_kws=dict(episodes=self.episodes, alpha=alpha))        

    def test3(self):
        alpha = 0.01
        np.random.seed(0)
        eps = EpsDecayGreedy(self.env.actions, 100)
        agent = QLearning(self.env, 1.0, HashTransformer(), eps, alpha)
        uc.learnQV(agent,
                   self.episodes,
                   self.env,
                   'QLearning',
                   save_Q=False,
                   save_V=False,
                   algo_kws=dict(episodes=self.episodes, alpha=alpha))        

    def test4(self):
        alpha = 0.1
        np.random.seed(0)
        eps = EpsDecayGreedy(self.env.actions, 100)
        agent = DoubleQLearning(self.env, 1.0, HashTransformer(), eps, alpha )
        uc.learnQV(agent, 
                   self.episodes,
                   self.env,
                   'DoubleQLearning',
                   save_Q=False,
                   save_V=False,
                   algo_kws=dict(episodes=self.episodes, alpha=alpha))
        
    def test6(self):
        np.random.seed(0)
        eps = EpsDecayGreedy(self.env.actions, 100)
        agent = SARSA(self.env, 1.0, HashTransformer(), eps, 0.01)
        episodes = 100000        
        with open('results/dumps/QMC_1000000.pickle', 'rb') as f:
            target_Q = pickle.load(f)

        cz = lambda : np.zeros(self.env.action_space.n)
        target_Q_ = defaultdict(cz)
        for i in range(target_Q.shape[0]):
            for j in range(target_Q.shape[1]):
                k = np.array([i+1, j+1], dtype=int)
                target_Q_[agent.transformer.transform(k)] = target_Q_[i,j]
        y = uc.RMSE(agent, episodes, self.env, "SARSA", target_Q_)
        plot_learning_curve_(np.arange(0, episodes), y)