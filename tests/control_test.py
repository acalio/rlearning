import unittest
from utils import EnvFactory
from agent.mc.mc_control import MCControlAgent
from utils.hash_transformer import HashTransformer
from utils.plots import plot_V, plot_learning_curve
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle 

class Test(unittest.TestCase):
    def setUp(self):
        self.env = EnvFactory.getEnvironment('easy21-v0',0)
        np.random.seed(0)
        self.agent = MCControlAgent(1.0, self.env, 1.0, HashTransformer())

    def test1(self):
        player, dealer = self.env.observation_space.high
        print(player, dealer)
        episodes = 50000
        info = self.agent.learn(episodes)
        transf = self.agent.transformer
        V = self.agent.get_state_value_function()
        V_matrix = np.zeros((player, dealer))
        for k,v in V.items():
            state = transf.invtransform(k) - 1
            V_matrix[tuple(state.astype(int))] = v
        plot_V(V_matrix, dealer, player)
