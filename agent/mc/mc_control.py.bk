from agent import ControlAgent, ApproximationAgent
import numpy as np
from collections import defaultdict
from utils import printProgressBar

class MCControlAgent(ControlAgent):
    """
    On-Policy first visit MC control (for \eps - soft policies)
    """

    def __init__(self, env, discount_factor, transformer, policy, every_visit=False):
        super().__init__(env, discount_factor, transformer, policy)
        zeros = lambda : np.zeros(self.env.action_space.n)
        self.state_action_visit = defaultdict(zeros)
        self.every_visit = every_visit

    def learn(self, episodes):
        """
        Implementation of the on-policy first visit
        MC control algorithm with eps-soft policy algorithm.
        page 101 of the Sutt7on and Barto Book
        """
        printProgressBar(0, episodes, prefix = 'Learning:', suffix = 'Complete', length = 50)
        
        # statistics anout the learning process
        info = {
            'rpe': []
        }

        for e in range(episodes):
            if (e+1)%100 == 0:
                printProgressBar(e+1, episodes, prefix = 'Learning:', suffix = 'Complete', length = 50)
            
            states, actions, rewards = self.generate_episode(as_separate_array=True,\
                state_dim=self.env.observation_space.shape)
            
            greturn = 0
            for i in range(len(states)):
                s,a,r = states[-i-1], int(actions[-i-1]), rewards[-i-1]
                greturn = self._return(greturn, r,a)
                if self.every_visit or self._is_first_visit(states[-i-1], states[:-i-1]):
                    self._update(greturn, s, a)
            #save the cumulative reward
            info['rpe'].append(greturn)

        return info


    def select_action(self, state):
        state = self.transformer.transform(state)
        return self.policy(state, self._Q[state])


    def _is_first_visit(self, state, previous_states):
        '''check if the agent is in state for the first time'''
        for s in previous_states:
            if np.all(state==s):
                return False
        return True
    
    def _update(self, greturn, state, action):
        Q, state_action_visit = self._Q, self.state_action_visit
        state = self.transformer.transform(state)
        state_action_visit[state][action] += 1
        Q[state][action] += 1/state_action_visit[state][action] * (greturn - Q[state][action])
    
    def _return(self, greturn, reward, action):
        return self.discount_factor*greturn + reward

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

        super().__init__(env, discount_factor, transformer, target_policy)
        self.b_policy = behavorial_policy
        self.every_visit = every_visit
        self.regular_is = ratio == "regular"
        
        

    def learn(self, episodes):
        """
        Implementation of the off-policy  MC control
        algorithm on page 111 of the Sutton and Barto Book
        """
        printProgressBar(0, episodes, prefix = 'Learning:', suffix = 'Complete', length = 50)

        # statistics anout the learning process
        info = {
            'rpe': []
        }

        for e in range(episodes):
            if (e+1)%100 == 0:
                printProgressBar(e+1, episodes, prefix = 'Learning:', suffix = 'Complete', length = 50)
            
            states, actions, rewards = self.generate_episode(as_separate_array=True,\
                state_dim=self.env.observation_space.shape)
            
            greturn = 0
            is_ratio = 1
            Q = self._Q
            for i in range(len(states)):
                s,a,r = states[-i-1], int(actions[-i-1]), rewards[-i-1]
                transf_s = self.transformer.transform(s)
                #scale the return with the importance sampling ratio
                greturn = self._return(greturn, a, r)

                is_ratio *= self.policy.get_actions_probabilities(s, Q[transf_s])[a]\
                                / self.b_policy.get_actions_probabilities(s, Q[transf_s])[a]

                if self.every_visit or self._is_first_visit(states[-i-1], states[:-i-1]):
                    scaled_greturn = is_ratio * greturn
                    self._update(scaled_greturn, states[-i-1], int(actions[-i-1]))
            
            #save the cumulative reward
            info['rpe'].append(greturn)

        return info

            
    def select_action(self, state):
        state = self.transformer.transform(state)
        return self.b_policy(state, self._Q[state])
    
        

class MCControlFA(MCControlAgent, ApproximationAgent):

    def __init__(self, env, discount_factor, transformer, policy, estimator, feature_converter, learning_rate, every_visit=False):
        ApproximationAgent.__init__(self, env, discount_factor, transformer, policy, estimator, feature_converter, learning_rate)        
        MCControlAgent.__init__(self, env, discount_factor, transformer, policy, every_visit)


    def select_action(self, state):
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
    