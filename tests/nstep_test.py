import unittest
from utils import EnvFactory
from agent.nsteps.nstep_control import nStepSarsa, nStepTreeBackup
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
        
    def test1(self):
        np.random.seed(0)
        eps = EpsDecayGreedy(self.env.actions, 100)
        agent = nStepSarsa(self.env, 1.0, HashTransformer(), eps, 0.01, 2)
        uc.learnQV(agent, 500000, self.env, 'nstep1')   
        uc.play(agent, self.env, 1000)


    def test2(self):
        np.random.seed(0)
        eps = EpsDecayGreedy(self.env.actions, 100)
        agent = nStepTreeBackup(self.env, 1.0, HashTransformer(), eps, 0.01, 2)
        uc.learnQV(agent, 100000 , self.env, 'ntreebackup')   
        uc.play(agent, self.env, 1000)


