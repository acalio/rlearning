import unittest
from utils import EnvFactory
from agent.nsteps.nstep_prediction import nStepPrediction
from utils.hash_transformer import HashTransformer
from utils.plots import plot_V
import numpy as np
import tests.usecases as uc
from agent.policy import Random


class PredictionTest(unittest.TestCase):
    def setUp(self):
        self.env = EnvFactory.getEnvironment('easy21-v0')
        
    def test3(self):
        agent = nStepPrediction(self.env, 1.0, HashTransformer(),Random(self.env.actions), 0.01, 2)
        uc.learnQV(agent,100000, self.env, 'step_prediction')

