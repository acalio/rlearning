from agent.tdlearning import TDAgent
from agent import PredictionAgent
from collections import defaultdict
from utils import printProgressBar
import numpy as np

class TDPrediction(PredictionAgent,TDAgent):
    """
    On-Policy TD-Learning for prediction
    """

    def __init__(self, env, discount_factor, transformer, policy, alpha):
        """Constructor
        
        Parameters
        ----------
            env : openAI environment
            disocunt_factor : float
            transformer : StateTransformer
            policy : Policy, to evaluate
            alpha : float, step size of the update rule

        """
        PredictionAgent.__init__(self, env, discount_factor, transformer, policy)
        TDAgent.__init__(self, env, discount_factor, transformer, policy, alpha)
        

    def select_action(self, state):
        return self.policy()

    def get_state_value_function(self, **kwargs):
        """
        Return the state value funciton
        """
        return self.V   

    def _td_error(self, state, action, reward, next_state):
        if next_state is None:
            next_state_value = 0
        else:
            next_state_value = self.V[state]

        current_state_value = self.V[state]
        return reward + self.discount_factor * next_state_value - current_state_value

    def _update_estimate(self, state, action, td_error):
        self.V[state] += self.alpha * td_error
