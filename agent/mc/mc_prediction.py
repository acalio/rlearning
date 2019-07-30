from agent import PredictionAgent
from utils import EnvFactory, RandomGenerator, printProgressBar
import numpy as np
from collections import defaultdict


class MCPredictionAgent(PredictionAgent):
    """
    First-Visit MC prediction  
    """

    def __init__(self, env, discount_factor, transformer):
        super().__init__(env, discount_factor, transformer)

        # initialize data structures
        self.state_shape = self.env.observation_space.shape
        self.state_visits = defaultdict(float)


    def learn(self, episodes):
        """
        Implementation of the first-visit MC prediction
        algorithm on page 92 of the Sutton and Barto book
        """
        printProgressBar(0, episodes, prefix = 'Learning:', suffix = 'Complete', length = 50)
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
                    self.state_visits[state] += 1
                    self.V[state] += (greturn-self.V[state])/self.state_visits[state]

    def select_action(self, state):
        return self.env.action_space.sample()


    def __is_first_visit(self, state, previous_states):
        '''check if the agent is in state for the first time'''
        for s in previous_states:
            if np.all(state==s):
                return False
        return True


