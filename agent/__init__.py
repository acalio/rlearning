import numpy as np
from utils.state_transformer import StateTransformer
from agent.policy import Policy
from abc import ABC, abstractmethod
from copy import copy
from collections import defaultdict


class Agent(ABC):
    '''Base class for every learning agent'''

    def __init__(self, env, discount_factor, transformer, policy):
        """ Constructor         
        Parameters
        -----------
            env : openai gym environment
            discount_factor : float
        """
        self.env = env
        self.discount_factor = discount_factor
        self.transformer = transformer
        self.policy = policy

    @abstractmethod
    def learn(self, episodes):
        """ Method to override with the learning algorithm

        Parameters
        ----------
            episode : int, number of episodes
        """        
        pass

    @abstractmethod
    def select_action(self, state):
        """ Select an action given a state, according to a
            given policy
        
        Parameters
        ----------
            state : current state of the environment
        Return
        ------
            an action
        """
        pass

    @abstractmethod
    def get_state_value_function(self, **kwargs):
        """Return the value function"""
        pass

    def generate_episode(self, as_separate_array = False, state_dim = None):
        """ Generate an episode following a given policy

        Parameters
        ----------
            as_separate_array : bool, wheter or not the episode should 
                                be return as three seperate array.
            state_dim : int, state's dimension
        Return
        ------
            an entire trajectory. Either in the form of
            a list of tuple <state, action, reward> or in the form
            of three separate arrays
        """
        episode = []
        done = False
        observation = self.env.reset()
        while not done:
            action = self.select_action(self.transformer.transform(observation))
            next_observation, reward, done, _ = self.env.step(action)
            episode.append((copy(observation), action, reward))
            observation = copy(next_observation)

        if as_separate_array:
            states = np.tile(np.zeros(state_dim, dtype=self.env.observation_space.dtype), [len(episode),1])
            actions = np.zeros(len(episode))
            rewards = np.zeros(len(episode))
            for i,(o,a,r) in enumerate(episode):
                states[i], actions[i], rewards[i] = o,a,r
            return states, actions, rewards
        return episode



class PredictionAgent(Agent):
    def __init__(self, env, discount_factor, transformer, policy):
        Agent.__init__(self, env, discount_factor, transformer, policy)
        self.V = defaultdict(float)

    def get_state_value_function(self, **kwargs):
        return self.V
    



class ControlAgent(Agent):

    def __init__(self, env, discount_factor, transformer, policy):
        Agent.__init__(self, env, discount_factor, transformer, policy)
        zeros = lambda : np.zeros(self.env.action_space.n)
        self.Q = defaultdict(zeros)

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
        num_actions = self.env.action_space.n
        cz = lambda : np.zeros(num_actions)
        V = defaultdict(cz)
        for k, action_array in self.Q.items():
            if optimality:
                V[k] = np.max(action_array)
            else:
                max_action = np.argmax(action_array)
                probs = self.policy.get_actions_probabilities(k, Q)
                mean = np.dot(action_array, probs)
                V[k] = mean
        return V