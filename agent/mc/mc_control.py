from agent import Agent
import numpy as np
from collections import defaultdict
from utils import printProgressBar


class MCControlAgent(Agent):
    """
    On-Policy first visit MC control (for \eps - soft policies)
    """

    def __init__(self, env, discount_factor, transformer, policy, every_visit=False):
        super().__init__(env, discount_factor, transformer, policy)
        
        self.Q = defaultdict(self._callable_zeros)
        self.state_action_visit = defaultdict(self._callable_zeros)
        self.every_visit = every_visit
        self.state_visit = defaultdict(int)

    def learn(self, episodes):
        """
        Implementation of the on-policy first visit
        MC control algorithm with eps-soft policy algorithm.
        page 101 of the Sutton and Barto Book
        """
        printProgressBar(0, episodes, prefix = 'Learning:', suffix = 'Complete', length = 50)
        
        # statistics anout the learning process
        info = {
            'rpe': []
        }

        for e in range(episodes):
            if (e+1)%100 == 0:
                pass #printProgressBar(e+1, episodes, prefix = 'Learning:', suffix = 'Complete', length = 50)
            
            states, actions, rewards = self.generate_episode(as_separate_array=True,\
                state_dim=self.env.observation_space.shape)
            
            greturn = 0
            for i in range(len(states)):
                greturn = self.discount_factor*greturn + rewards[-i-1]
                if self.every_visit or self._is_first_visit(states[-i-1], states[:-i-1]):
                    state = self.transformer.transform(states[-i-1])
                    action = int(actions[-i-1])
                    self.state_action_visit[state][action] += 1
                    self.Q[state][action] += 1 / self.state_action_visit[state][action] * (greturn - self.Q[state][action])
                    self.state_visit[state] += 1

            #save the cumulative reward
            info['rpe'].append(greturn)

        return info


    def select_action(self, state):
        visit = self.state_visit[state]
        self.policy.update(state_visit = visit)
        return self.policy(state, self.Q)


    def get_state_value_function(self, **kwargs):
        """
        Get the state value from the Q-values derived
        from the learning process. The Vs can be either
        selected by taking the expectation or the max

        Parameters
        ----------
            optimality : boolean,  true if for each state-action pair
                         you want to select the maximal value
        
        Returns
        -------
            State values for every state encounter in the learning phase
        """
        try:
            optimality = kwargs['optimality']
        except KeyError:
            optimality = False

        V = defaultdict(self._callable_zeros)
        num_actions = self.env.action_space.n
        for k, action_array in self.Q.items():
            if optimality:
                V[k] = np.max(action_array)
            else:
                max_action = np.argmax(action_array)
                probs = self.policy.get_actions_probabilities(k, Q)
                mean = np.dot(action_array, probs)
                V[k] = mean
        return V


    def _callable_zeros(self):
        '''Utility function for reating an array of all zero'''
        return np.zeros(self.env.action_space.n)


    def _is_first_visit(self, state, previous_states):
        '''check if the agent is in state for the first time'''
        for s in previous_states:
            if np.all(state==s):
                return False
        return True
    


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
        
        # init data structure
        self.Q = defaultdict(self._callable_zeros)
        self.state_action_visit = defaultdict(self._callable_zeros)
        

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
            for i in range(len(states)):
                state_transformed = self.transformer.transform(states[-i-1])
                action = int(actions[-i-1])

                #scale the return with the importance sampling ratio
                greturn = self.discount_factor*greturn + rewards[-i-1]
                is_ratio *= self.policy.get_actions_probabilities(state_transformed, self.Q)[action]\
                                / self.b_policy.get_actions_probabilities(state_transformed, self.Q)[action]

                if self.every_visit or self._is_first_visit(states[-i-1], states[:-i-1]):
                    scaled_greturn = is_ratio * greturn
                    self.state_action_visit[state_transformed][action] += 1
                    self.Q[state_transformed][action] += (greturn-self.Q[state_transformed][action])\
                        /self.state_action_visit[state_transformed][action]
            
            #save the cumulative reward
            info['rpe'].append(greturn)

        return info

            
    def select_action(self, state):
        return self.b_policy(state, self.Q)