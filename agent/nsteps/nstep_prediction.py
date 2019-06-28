from agent import PredictionAgent
from agent.nsteps import nStepAgent
from collections import defaultdict, deque
import numpy as np


class nStepPrediction(nStepAgent, PredictionAgent):
    """
    On-Policy nstep prediction
    """
    def __init__(self, env, discount_factor, transformer, policy, alpha, nsteps):
        PredictionAgent.__init__(self, env, discount_factor, transformer, policy)
        nStepAgent.__init__(self,env, discount_factor, transformer, policy, alpha, nsteps)
        self.state_visit = defaultdict(int)
        self.gammas = np.array([np.power(self.discount_factor, -i) for i in range(self.nsteps+1)])

    def _get_target(self):
        rewards = [r for _, _ ,r in self.q]
        rlen = len(rewards)
        if rlen == self.nsteps:
            '''
            it means the agent is not in a terminal state
            therefore the last reward must be replaced with the q value
            of the last state action
            '''
            last_s, _ , _ = self.q[-1]
            rewards[-1] = self.V[last_s]

        return np.dot(self.gammas[:rlen], rewards)

    def _update_estimate(self, target):
        s, a, r = self.q.pop()
        self.state_visit[s] += 1
        self.alpha = 1/self.state_visit[s]
        self.V[s] += self.alpha * (target - self.V[s]) 

    def select_action(self, state):
        return self.policy(state)
