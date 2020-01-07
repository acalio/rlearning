from agent.tdlearning import TDAgent
from agent import PredictionAgent, ApproximationAgent
from collections import defaultdict
from utils import printProgressBar
from agent.policy import Random
import numpy as np

class TDPrediction(PredictionAgent,TDAgent):
    """
    On-Policy TD-Learning for prediction
    """

    def __init__(self, env, discount_factor, transformer, alpha):
        """Constructor
        
        Parameters
        ----------
            env : openAI environment
            disocunt_factor : float
            transformer : StateTransformer
            policy : Policy, to evaluate
            alpha : float, step size of the update rule

        """
        PredictionAgent.__init__(self, env, discount_factor, transformer, Random(env.actions))
        TDAgent.__init__(self, env, discount_factor, transformer, Random(env.actions), alpha)
        

    def select_action(self, state):
        return self.policy()

    def get_state_value_function(self, **kwargs):
        """
        Return the state value funciton
        """
        return self._V   

    def _td_error(self, state, action, reward, next_state, next_action):
        state = self.transformer.transform(state)
        if next_state is not None:
            next_state = self.transformer.transform(next_state)
            next_state_value = self._V[next_state]
        else:
            next_state_value = 0

        current_state_value = self._V[state]

        return reward + self.discount_factor * next_state_value - current_state_value

    def _update(self, state, action, td_error):
        state = self.transformer.transform(state)
        self._V[state] += self.alpha * td_error

    

class TDPredictionFA(TDPrediction, ApproximationAgent):

    def __init__(self, env, discount_factor, transformer, estimator, feature_converter, learning_rate):
        ApproximationAgent.__init__(self, env, discount_factor, transformer, Random(env.actions), estimator, feature_converter, learning_rate)
        TDPrediction.__init__(self, env, discount_factor, transformer, learning_rate)
        self.state_visits = defaultdict(int)

    def _td_error(self, state, action, reward, next_state, next_action):
        feat_state = self.feature_converter.transform(state)
        if next_state is None:
            estimate_next_state = 0
        else:
            feat_next_state = self.feature_converter.transform(next_state)
            estimate_next_state = self.estimator(feat_next_state)

        return reward + self.discount_factor * estimate_next_state - self.estimator(feat_state)

    def select_action(self, state):
        return self.policy(state)

    def _update(self, state, action, td_error):
        feat_state = self.feature_converter.transform(state)
        trans_feat_state = self.transformer.transform(feat_state)
        self.state_visits[trans_feat_state] += 1
        self._update_estimator(td_error, feat_state)