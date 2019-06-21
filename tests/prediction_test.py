import unittest
from utils import EnvFactory
from agent.mc.mc_prediction import MCPredictionAgent
from utils.hash_transformer import HashTransformer
from utils.plots import plot_V
import numpy as np

class PredictionTest(unittest.TestCase):
    def setUp(self):
        self.env = EnvFactory.getEnvironment('easy21-v0')
        self.agent = MCPredictionAgent(self.env, 1.0, HashTransformer())
        
    def test3(self):
        player_range, dealer_range = self.env.observation_space.high+1
        self.agent.learn(50000)
        
        transf = self.agent.transformer
        V = self.agent.get_state_value_function()
        V_matrix = np.zeros((player_range, dealer_range))
        for k,v in V.items():
            state = transf.invtransform(k)
            V_matrix[tuple(state.astype(int))] = v
        plot_V(V_matrix, player_range, dealer_range)
