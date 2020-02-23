from agent import PredictionAgent, ApproximationAgent
from utils import EnvFactory, RandomGenerator, printProgressBar
import numpy as np
from collections import defaultdict
from agent.mc import MCAgent

class MCPredictionAgent(MCAgent, PredictionAgent):
    """
    First-Visit MC prediction  
    """

    def __init__(self, env, discount_factor, transformer, policy, every_visit = False):
        MCAgent.__init__(self, env, discount_factor, transformer, policy, every_visit)
        PredictionAgent.__init__(self, env, discount_factor, transformer, policy)
        # initialize data structures
        self.state_shape = self.env.observation_space.shape
        self.state_visits = defaultdict(float)
        # self._V = defaultdict(float)


    def _return(self, s, action, reward, curr_return):
        return self.discount_factor*curr_return + reward

    def _update(self, greturn, state, action):
        state = self.transformer.transform(state)
        self.state_visits[state] += 1
        self._V[state] += (greturn-self._V[state])/self.state_visits[state]

    def get_state_value_function(self, **kwargs):
        return self._V

class MCPredictionFA(MCPredictionAgent, ApproximationAgent):
    """
    Monte Carlo control with function approximation
    """

    def __init__(self, env, discount_factor, transformer, policy, estimator, feature_converter, learning_rate, every_visit=False):
        ApproximationAgent.__init__(self, env, discount_factor, transformer, policy, estimator, feature_converter, learning_rate)        
        MCPredictionAgent.__init__(self, env, discount_factor, transformer, policy, every_visit)
        

    def select_action(self, state):
        #create aliases
        featurize = self.feature_converter.transform
        estimate = self.estimator

        values = estimate(featurize(state))
        return self.policy(state, values)

    def _update(self, greturn, state, action):
        state = self.transformer.transform(state)
        predicted = self.estimator.predict(state)
        self.estimator.compute_loss(greturn, predicted)
        self.estimator.update_weights()
    
    def get_state_value_function(self, **kwargs):
        pass
