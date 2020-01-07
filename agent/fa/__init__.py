from utils import printProgressBar
from agent import Agent
from copy import copy
from abc import abstractmethod
from collections import deque
import numpy as np

class ApproximationAgent(Agent):

    def __init__(self, env, discount_factor, transformer, policy, estimator, feature_converter):
        Agent.__init__(self,env, discount_factor, transformer, policy)
        self.alpha = 0
        self.estimator = estimator
        self.feature_converter = feature_converter
        

                        
    def _update_estimate(self, error, state_feature):
        self.estimator.w += self.alpha * error * self.estimator.get_derivative()(state_feature)
        
    