
from utils import printProgressBar
from agent import Agent
from copy import copy
from abc import abstractmethod

class TDAgent(Agent):
    def __init__(self, env, discount_factor, transformer, policy, alpha):
        Agent.__init__(self, env, discount_factor, transformer, policy)
        self.alpha = alpha

    def learn(self, episodes):
        printProgressBar(0, episodes, prefix = 'Learning:', suffix = 'Complete', length = 50)
        
        # create aliases
        select_action = self.select_action       
        reset = self.env.reset
        step = self.env.step
        td_error = self._td_error


        # statistics anout the learning process
        info = {
            'rpe': []
        }

        for e in range(episodes):
            
            if (e+1)%100 == 0:
                printProgressBar(e+1, episodes, prefix = 'Learning:', suffix = 'Complete', length = 50)
            
            done = False
            observation = reset()
            next_action = select_action(observation)
            while not done:
                action = next_action
                next_observation, reward, done, _ = step(action)

                if done:
                    next_observation, next_action = None, None
                else:
                    next_action = select_action(next_observation)

                td_error = self._td_error(observation, action, reward, next_observation, next_action)
                self._update(observation, action, td_error)
                observation = copy(next_observation)

        return info

        
    @abstractmethod
    def _td_error(self, state, action, reward, next_state, next_action):
        """
        Compute the TD-Error
       
        Parameters:
        -----------
            state : current_state
            action : current_action
            reward : reward due to action a in state s
            next_state : state next to state
            next_action : action next_to
        """
        pass

    @abstractmethod
    def _update(self, state, action, td_error):
        pass
    

