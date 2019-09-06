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
import tests.usecases as uc

RESULT_PATH = './results'

class Test(unittest.TestCase):
    def setUp(self):
        self.env = EnvFactory.getEnvironment('easy21-v0',0)
        np.random.seed(0)
        
        

    def test1(self):
        episodes = 10000
        eps = EpsDecayGreedy(self.env.actions, 1000)
        agent = MCControlAgent(self.env, 1.0, HashTransformer(), eps, every_visit=True)
        uc.learnQV(agent, episodes, 
                   self.env,
                   'mc',
                   save_Q=False,
                   save_V=False,
                   algo_kws=dict(episodes=episodes))
