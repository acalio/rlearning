import unittest
from utils import EnvFactory
from agent.tdlearning.td_prediction import TDPrediction
from utils.hash_transformer import HashTransformer
from agent.policy import Random, EpsGreedy
from utils.plots import plot_V
import numpy as np

class PredictionTest(unittest.TestCase):
    def setUp(self):
        self.env = EnvFactory.getEnvironment('easy21-v0')
        eps = Random(self.env.actions)
        self.agent = TDPrediction(self.env, 1.0, HashTransformer(), 0.0001)
        
    def test3(self):
        player, dealer = self.env.observation_space.high
        self.agent.learn(100000)
        transf = self.agent.transformer
        V = self.agent.get_state_value_function()
        V_matrix = np.zeros((player, dealer))
        for k,v in V.items():
            state = transf.invtransform(k) - 1
            V_matrix[tuple(state.astype(int))] = v
        plot_V(V_matrix, dealer, player)#, save="mc.pdf")