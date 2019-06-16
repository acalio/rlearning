import numpy as np
from utils.state_transformer import StateTransformer
from agent.policy import Policy
from abc import ABC, abstractmethod
from copy import copy

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
    def get_state_value_function(self):
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
            states = np.tile(np.zeros(state_dim), [len(episode),1])
            actions = np.zeros(len(episode))
            rewards = np.zeros(len(episode))
            for i,(o,a,r) in enumerate(episode):
                states[i], actions[i], rewards[i] = o,a,r
            return states, actions, rewards
        return episode

