from agent import Agent
import numpy as np
from collections import defaultdict
from utils import printProgressBar


class MCControlAgent(Agent):
    """
    On-Policy first visit MC control (for \eps - soft policies)
    """

    def __init__(self, eps, env, discount_factor, transformer, policy = "eps-greedy"):
        super().__init__(env, discount_factor, transformer)
        
        self.Q = defaultdict(self.__callable_zeros)
        self.state_action_visit = defaultdict(self.__callable_zeros)
        self.eps = eps 
        if policy == 'eps-greedy':
            self.__select = self.__eps_greedy_selection
        else:
            self.__select = self.__eps_soft_selection


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

        for i in range(episodes):
            if (i+1)%100 == 0:
                printProgressBar(i+1, episodes, prefix = 'Learning:', suffix = 'Complete', length = 50)
            
            states, actions, rewards = self.generate_episode(as_separate_array=True,\
                state_dim=self.env.observation_space.shape)
            
            greturn = 0
            for i in range(len(states)):
                greturn = self.discount_factor*greturn + rewards[-i-1]
                if self.__is_first_visit(states[-i-1], states[:-i-1]):
                    state = self.transformer.transform(states[-i-1])
                    action = int(actions[-i-1])
                    self.state_action_visit[state][action] += 1
                    self.Q[state][action] += (greturn-self.Q[state][action])/self.state_action_visit[state][action]
            
            #save the cumulative reward
            info['rpe'].append(greturn)

        return info


    def select_action(self, state):
        return self.__select(state)


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

        V = defaultdict(self.__callable_zeros)
        num_actions = self.env.action_space.n
        for k, action_array in self.Q.items():
            if optimality:
                V[k] = np.max(action_array)
            else:
                max_action = np.argmax(action_array)
                mean = np.dot(action_array, 
                    np.where(action_array==max_action, 1.0-self.eps+self.eps/num_actions, self.eps/num_actions))
                V[k] = mean
        return V


    def __callable_zeros(self):
        '''Utility function for reating an array of all zero'''
        return np.zeros(self.env.action_space.n)


    def __eps_soft_selection(self, state):
        ''' eps-soft policy'''
        num_actions = self.env.action_space.n
        actions = self.env.actions
        max_action = np.argmax(self.Q[state])
        return np.random.choice(actions,\
            p=np.where(actions==max_action, 
                       1.0-self.eps+self.eps/num_actions, # prob. greedy action
                       self.eps/num_actions))             # prob. not greedy action


    def __eps_greedy_selection(self, state):
        '''eps-greedy selection'''
        if np.random.rand() <= self.eps:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.Q[state])


    def __is_first_visit(self, state, previous_states):
        '''check if the agent is in state for the first time'''
        for s in previous_states:
            if np.all(state==s):
                return False
        return True
    