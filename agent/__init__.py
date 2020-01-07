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
    def learn(self, num_of_episodes):
        """ Method to override with the learning algorithm

        Parameters
        ----------
            num_of_episodes : int, number of episodes
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

    def generate_episode(self, as_separate_array = False):
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
        state_dim = self.env.observation_space.shape
        while not done:
            action = self.select_action(observation)
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

    def _start_episode(self):
        """ Callback to be called when at
        the beginning of a new episode
        """
        pass

    def _update(self, greturn, state, action):
        """ Update the estimates realted
        to the given state and action pair, 
        accordingly to the return

        Parameters
        ----------
            greturn : current episode return
            state : current state
            action : current action
        """
        pass



class ApproximationAgent(Agent):

    def __init__(self, env, discount_factor, transformer, policy, estimator, feature_converter, learning_rate = 0.001):
        Agent.__init__(self,env, discount_factor, transformer, policy)
        self.alpha = learning_rate
        self.estimator = estimator
        self.feature_converter = feature_converter
        
    def _get_target(self):
        pass
                        
    def _update_estimator(self, error, state_feature):
        self.estimator.w += self.alpha * error * self.estimator.get_derivative()(state_feature)
        if np.all(np.isnan(self.estimator.w)):
            raise RuntimeError("The weights got NaN! You may want to decrease the learning rate")
        


class PredictionAgent(Agent):
    def __init__(self, env, discount_factor, transformer, policy):
        Agent.__init__(self, env, discount_factor, transformer, policy)
        self._V = defaultdict(float)

    def get_state_value_function(self, **kwargs):
        return self._V
    



class ControlAgent(Agent):

    def __init__(self, env, discount_factor, transformer, policy):
        Agent.__init__(self, env, discount_factor, transformer, policy)
        zeros = lambda : np.zeros(self.env.action_space.n)
        self._Q = defaultdict(zeros)

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
        for k, action_array in self._Q.items():
            if optimality:
                V[k] = np.max(action_array)
            else:
                probs = self.policy.get_actions_probabilities(k, self._Q)
                mean = np.dot(action_array, probs)
                V[k] = mean
        return V



    