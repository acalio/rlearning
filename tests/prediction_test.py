import unittest
from utils import EnvFactory
from agent.mc.mc_prediction import MCPredictionAgent
from utils.hash_transformer import HashTransformer
from utils.plots import plot_V
from agent.policy import Random
import numpy as np
from tests import usecases as uc

class PredictionTest(unittest.TestCase):
    def setUp(self):
        self.env = EnvFactory.getEnvironment('easy21-v0')
        self.agent = MCPredictionAgent(self.env, 1.0, HashTransformer(), Random([0,1]))
        
    def test3(self):
        player_range, dealer_range = self.env.observation_space.high
        self.agent.learn(50000)
        V_matrix = uc.extract_V(self.agent, self.env)
        print(V_matrix)
        plot_V(V_matrix, dealer_range, player_range, save=None)