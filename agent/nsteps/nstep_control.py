from agent import ControlAgent
from agent.nsteps import nStepAgent
from collections import defaultdict, deque
import numpy as np


class nStepSarsa(nStepAgent, ControlAgent):
    """
    On-Policy nstep control
    """
    def __init__(self, env, discount_factor, transformer, policy, alpha, nsteps):
        ControlAgent.__init__(self, env, discount_factor, transformer, policy)
        nStepAgent.__init__(self,env, discount_factor, transformer, policy, alpha, nsteps)
        cz = lambda : np.zeros(self.env.action_space.n)
        self.state_action_visit = defaultdict(cz)
        self.gammas = np.array([np.power(self.discount_factor, -i) for i in range(self.nsteps+1)])

    def _get_target(self):
        rewards = [r for _, _ ,r in self.q]
        rlen = len(rewards)
        if rlen > self.nsteps:
            '''
            it means the agent is not in a terminal state
            therefore the last reward must be replaced with the q value
            of the last state action
            '''
            last_s, last_a, _ = self.q[-1]
            rewards[-1] = self.Q[last_s][last_a]

        return np.dot(self.gammas[:rlen], rewards)


    def _update_estimate(self, target):
        s, a, _ = self.q.popleft()
        self.state_action_visit[s][a] += 1
        self.alpha = 1/self.state_action_visit[s][a]
        self.Q[s][a] += self.alpha * (target - self.Q[s][a]) 


    def select_action(self, state):
        return self.policy(state, self.Q[state])


class nStepTreeBackup(nStepAgent, ControlAgent):
    """
    On-Policy nstep control
    """
    def __init__(self, env, discount_factor, transformer, policy, alpha, nsteps):
        ControlAgent.__init__(self, env, discount_factor, transformer, policy)
        nStepAgent.__init__(self,env, discount_factor, transformer, policy, alpha, nsteps)
        cz = lambda : np.zeros(self.env.action_space.n)
        self.state_action_visit = defaultdict(cz)
        self.gammas = np.array([np.power(self.discount_factor, -i) for i in range(self.nsteps+1)])

    def _get_target(self):
        # create refs for efficiency reasons
        rlen = len(self.q)
        q = self.q
        Q = self.Q
        get_actions_probs = self.policy.get_actions_probabilities
        gamma = self.discount_factor

        if rlen > self.nsteps:
            '''
            it means the agent is not in a terminal state
            therefore the last reward must be replaced with the expected q value
            in the last state of the sequence
            '''
            last_st, _, _ = q[-1]
            _ , _, last_r = q[-2]
            actions_qs = Q[last_st]
            g = last_r + gamma * np.dot(get_actions_probs(last_st, actions_qs), actions_qs)
        else:
            _, _ ,g = q[-1] # g is equal to the last reward (at the end of the episode)
        
        for i in range(2, rlen-1):
            s, a, _ = q[-i]
            _, _, r = q[-i-1]
            action_probs = get_actions_probs(s, Q[s])
            a_prob = action_probs[a]
            action_probs[a] = 0 # set to zero the action actually taken by the agent

            g = r + gamma * np.dot(action_probs, Q[s]) + gamma*a_prob*g

        return g


    def _update_estimate(self, target):
        s, a, r = self.q.popleft()
        self.state_action_visit[s][a] += 1
        self.alpha = 1/self.state_action_visit[s][a]
        self.Q[s][a] += self.alpha * (target - self.Q[s][a]) 


    def select_action(self, state):
        return self.policy(state, self.Q[state])


    

    
