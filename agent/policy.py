import numpy as np
from abc import ABC, abstractmethod 
from collections import defaultdict
from utils.hash_transformer import HashTransformer
from utils.dummy_transformer import DummyTransformer

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
    def __call__(self, state, action_values):
        pass

    @abstractmethod
    def get_actions_probabilities(self, action_values):
        pass

class EpsSoft(Policy):
    def __init__(self, actions, eps):
        super().__init__(actions)
        self.num_actions = len(actions)
        self.eps = eps

    def __call__(self, state, action_values):
        probs = self.get_actions_probabilities(state, action_values)
        return np.random.choice(self.actions, p=probs)

    def get_actions_probabilities(self, state, action_values):
        max_action = np.argmax(action_values)
        return np.where(self.actions==max_action,
                        1.0-self.eps+self.eps/self.num_actions, # prob. greedy action
                        self.eps/self.num_actions)


class EpsGreedy(Policy):
    def __init__(self, actions, eps):
        super().__init__(actions)
        self.num_actions = len(actions)
        self.eps = eps

        self.info = {'greedy': 0, 'games': 0}

    def __call__(self, state, action_values):
        self.info['games'] += 1
        if np.random.uniform() <= self.eps:
            return np.random.choice(self.actions)
        self.info['greedy'] += 1
        return self.actions[np.argmax(action_values)]

    def get_actions_probabilities(self, state, action_values):
        max_action = np.argmax(action_values)
        return np.where(self.actions==max_action,
                        1.0-self.eps+self.eps/self.num_actions, # prob. greedy action
                        self.eps/self.num_actions)
    

class EpsDecayGreedy(EpsGreedy):
    def __init__(self, actions, N = 100, transf = DummyTransformer()):
        super().__init__(actions, 1)
        self.state_visit = defaultdict(int)
        self.N = N
        self.transf = transf

    def __call__(self, state, action_values):
        state = self.transf.transform(state)
        self.state_visit[state] += 1
        self.eps = self.N / (self.N+self.state_visit[state])
        return EpsGreedy.__call__(self, state, action_values)


class Random(Policy):
    def __init__(self, actions):
        super().__init__(actions)
    
    def __call__(self, state = None, action_values = None):
        return np.random.choice(self.actions)

    def get_actions_probabilities(self, state, action_values = None):
        return np.repeat(1/len(self.actions), len(self.actions))