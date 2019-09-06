import unittest
from utils import EnvFactory
import numpy as np
from utils.features_transformer import LinearFeatureTransformer, PolynomialFeatureTransformer, TileFeatureTransformer
import estimators as e 
from agent.tdlearning.td_prediction import TDPredictionFA
from agent.mc.mc_prediction import MCPredictionFA
from agent.mc.mc_control import MCControlFA
from utils.hash_transformer import HashTransformer
from utils.dummy_transformer import DummyTransformer
from agent import policy as pol
from utils.plots import plot_V
import tests.usecases as uc

class Test(unittest.TestCase):
    def setUp(self):
        self.env = EnvFactory.getEnvironment('easy21-v0', 0)
        
    @unittest.skip
    def test1(self):
        rpolicy = pol.Random([0,1])
        transf = LinearFeatureTransformer(2,2)
        estimator = e.LinearEstimator(2)
        agent = MCPredictionFA(self.env, 1.0, HashTransformer(), rpolicy, estimator, transf)
        agent.learn(10000)
        player, dealer = self.env.observation_space.high
        V_matrix = np.zeros((player, dealer))
        for p in np.arange(0, player, 1):
            for d in np.arange(0, dealer):
                features = transf.transform(np.array([p,d]))
                V_matrix[int(p), int(d)] = estimator(features)
        plot_V(V_matrix, dealer, player)
        print(V_matrix)

    @unittest.skip
    def test2(self):
        rpolicy = pol.Random([0,1])
        transf = PolynomialFeatureTransformer(2,2)
        estimator = e.LinearEstimator(transf.transformed_shape[0])
        agent = MCPredictionFA(self.env, 1.0, HashTransformer(), estimator, transf, 0.00001)
        agent.learn(10000)
        player, dealer = self.env.observation_space.high
        V_matrix = np.zeros((player, dealer))
        for p in np.arange(0, player, 1):
            for d in np.arange(0, dealer):
                features = transf.transform(np.array([p,d]))
                V_matrix[int(p), int(d)] = estimator(features)
        plot_V(V_matrix, dealer, player)
        print(V_matrix)

    @unittest.skip
    def test3(self):
        rpolicy = pol.Random([0,1])
        transf = PolynomialFeatureTransformer(2,2)
        estimator = e.LinearEstimator(transf.transformed_shape[0])
        agent = TDPredictionFA(self.env, 1.0, HashTransformer(), estimator, transf, 0.00001)
        agent.learn(10000)
        player, dealer = self.env.observation_space.high
        V_matrix = np.zeros((player, dealer))
        for p in np.arange(0, player, 1):
            for d in np.arange(0, dealer):
                features = transf.transform(np.array([p,d]))
                V_matrix[int(p), int(d)] = estimator(features)
        plot_V(V_matrix, dealer, player)
        print(V_matrix)

    def test4(self):
        episodes = 10000
        transf = LinearFeatureTransformer(2,2)
        estimator = e.LinearEstimator(4)
        eps = pol.EpsDecayGreedy(np.arange(self.env.action_space.n), 1000, HashTransformer() )
        agent = MCControlFA(self.env, 1.0, HashTransformer(), eps, estimator, transf, 0.0001, every_visit=True)
        info = agent.learn(episodes)
        player, dealer = self.env.observation_space.high
        V_matrix = np.zeros((player, dealer))
        for p in np.arange(0, player, 1):
            for d in np.arange(0, dealer):
                values = [estimator(transf.transform(np.array([p,d]), action=i)) for i in range(self.env.action_space.n)]
                V_matrix[int(p), int(d)] = np.max(values)
        #plot_V(V_matrix, dealer, player)
        
        uc.play(agent, self.env, 100)
    
    