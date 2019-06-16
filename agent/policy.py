import numpy as np
from abc import ABC, abstractmethod 

class Policy(ABC):
    """
    Mapping from states and actions
    """
    def __init__(self, actions):
        """Constructor 
        
        Parameters
        ----------
            actions : a list of actions

        """
        self.actions = actions
    
    @abstractmethod
    def __call__(self, state, Q):
        pass

    @abstractmethod
    def get_actions_probabilities(self, state, Q):
        pass

class EpsSoft(Policy):
    def __init__(self, actions, eps):
        super().__init__(actions)
        self.num_actions = len(actions)
        self.eps = eps

    def __call__(self, state, Q):
        probs = self.get_actions_probabilities(state, Q)
        return np.random.choice(self.actions, p=probs)

    def get_actions_probabilities(self, state, Q):
        max_action = np.argmax(Q[state])
        return np.where(self.actions==max_action,
                        1.0-self.eps+self.eps/self.num_actions, # prob. greedy action
                        self.eps/self.num_actions)

class EpsGreedy(Policy):
    def __init__(self, actions, eps):
        super().__init__(actions)
        self.num_actions = len(actions)
        self.eps = eps

    def __call__(self, state, Q):
        if np.random.rand() <= self.eps:
            return np.random.choice(self.actions)
        else:
            return np.argmax(Q[state])

    def get_actions_probabilities(self, state, Q):
        max_action = np.argmax(Q[state])
        return np.where(self.actions==max_action,
                        1.0-self.eps+self.eps/self.num_actions, # prob. greedy action
                        self.eps/self.num_actions)

class Random(Policy):
    def __init__(self, actions):
        super().__init__(actions)
    
    def __call__(self, state, Q = None):
        return np.random.choice(self.actions)


    def get_actions_probabilities(self, state, Q):
        max_action = np.argmax(Q[state])
        return np.repeat(1/len(self.actions), len(self.actions))