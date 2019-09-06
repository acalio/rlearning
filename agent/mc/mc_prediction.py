from agent import PredictionAgent, ApproximationAgent
from utils import EnvFactory, RandomGenerator, printProgressBar
import numpy as np
from collections import defaultdict
from agent.policy import Random

class MCPredictionAgent(PredictionAgent):
    """
    First-Visit MC prediction  
    """

    def __init__(self, env, discount_factor, transformer):
        super().__init__(env, discount_factor, transformer, Random(np.arange(env.action_space.n)))

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
                    self._update(greturn, states[-i-1])

    def select_action(self, state):
        state_transformed = self.transformer.transform(state)
        return self.policy(state_transformed)


    def __is_first_visit(self, state, previous_states):
        '''check if the agent is in state for the first time'''
        for s in previous_states:
            if np.all(state==s):
                return False
        return True

    def _update(self, greturn, state):
        state = self.transformer.transform(state)
        self.state_visits[state] += 1
        self._V[state] += (greturn-self._V[state])/self.state_visits[state]



class MCPredictionFA(MCPredictionAgent,ApproximationAgent):
    
    def __init__(self, env, discount_factor, transformer, estimator, feature_converter, learning_rate):
        ApproximationAgent.__init__(self, env, discount_factor, transformer, Random(np.arange(env.action_space.n)), estimator, feature_converter, learning_rate)
        MCPredictionAgent.__init__(self, env, discount_factor, transformer)
                

    def select_action(self, state):
        state_feature = self.transformer.transform(state)
        return self.policy(state_feature)
    
    def get_state_value_function(self, **kwargs):
        pass

    def _update(self, greturn, state):
        state_feature = self.feature_converter.transform(state)
        error = greturn - self.estimator(state_feature)
        self._update_estimator(error, state_feature)
        