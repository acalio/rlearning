from agent import ControlAgent, ApproximationAgent
from agent.mc import MCAgent
import numpy as np
from collections import defaultdict
from utils import printProgressBar

class MCControlAgent(MCAgent,ControlAgent):
    """
    On-Policy first visit MC control (for \eps - soft policies)
    """

    def __init__(self, env, discount_factor, transformer, policy, every_visit=False):
        MCAgent.__init__(self, env, discount_factor, transformer, policy, every_visit)
        ControlAgent.__init__(self, env, discount_factor, transformer, policy)

        self.state_action_visit = defaultdict(lambda : np.zeros(self.env.action_space.n))


    def _update(self, greturn, state, action):
        Q, state_action_visit = self._Q, self.state_action_visit
        state = self.transformer.transform(state)
        state_action_visit[state][action] += 1
        Q[state][action] += 1/state_action_visit[state][action] * (greturn - Q[state][action])
    
    def select_action(self, state):
        state_transformed = self.transformer.transform(state)
        return self.policy(state_transformed, self._Q[state_transformed])

    def _return(self, s, action, reward, curr_return):
        return self.discount_factor*curr_return + reward


class OffPolicyMCControlAgent(MCControlAgent):
    """
    Off-Policy MonteCarlo control via Important Sampling
    """

    def __init__(self, env, discount_factor, transformer, target_policy, behavorial_policy, every_visit = False, ratio = "regular"):
        """Constructor
        
        Parameters
        ----------
            env : an environment
            discount_factor : float
            transformer : a StateTransformer
            target_policy : the policy to lear
            behavorial_policy : the policy to follow
            every_visit : boolean
            ratio : ["regular", "weighted"]
        """

        MCControlAgent.__init__(self, env, discount_factor, transformer, target_policy)
        self.b_policy = behavorial_policy
        self.every_visit = every_visit
        self.regular_is = ratio == "regular"
        self.current_episode = []
        self.is_ratio = 1
        

    def select_action(self, state):
        tstate = self.transformer.transform(state)
        action = self.b_policy(state, self._Q[tstate])

        is_ratio = self.policy.get_actions_probabilities(tstate, self._Q[tstate])[action]
        is_ratio /= self.b_policy.get_actions_probabilities(tstate, self._Q[tstate])[action]

        self.current_episode.append(is_ratio)

        return action

    def _return(self, state, action, reward, curr_return):
        self.is_ratio *= self.current_episode.pop()
        return self.is_ratio / self.discount_factor*curr_return + reward    
    
    def _update(self, greturn, state, action):
        Q, state_action_visit = self._Q, self.state_action_visit
        state = self.transformer.transform(state)
        state_action_visit[state][action] += 1
        Q[state][action] += 1/state_action_visit[state][action] * (greturn - Q[state][action])
    

    def _start_episode(self):
        self.current_episode.clear()
        self.is_ratio = 1


class MCControlFA(MCControlAgent, ApproximationAgent):
    """
    Monte Carlo control with function approximation
    """

    def __init__(self, env, discount_factor, transformer, policy, estimator, feature_converter, learning_rate, every_visit=False):
        ApproximationAgent.__init__(self, env, discount_factor, transformer, policy, estimator, feature_converter, learning_rate)        
        MCControlAgent.__init__(self, env, discount_factor, transformer, policy, every_visit)


    def select_action(self, state):
        #create aliases
        featurize = self.feature_converter.transform
        estimate = self.estimator

        values = [estimate(featurize(state, action=i)) for i in range(self.env.action_space.n)]
        return self.policy(state, values)

    def _update(self, greturn, state, action):
        state_action_feature = self.feature_converter.transform(state, action = action)
        state_action_transformed = self.transformer.transform(state_action_feature)
        self.state_action_visit[state_action_transformed] += 1
        error = greturn - self.estimator(state_action_feature)
        self._update_estimator(error, state_action_feature)
    