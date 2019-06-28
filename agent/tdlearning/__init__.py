
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
        
        # statistics anout the learning process
        info = {
            'rpe': []
        }

        for e in range(episodes):
            if (e+1)%100 == 0:
                printProgressBar(e+1, episodes, prefix = 'Learning:', suffix = 'Complete', length = 50)
            
            done = False
            observation = self.env.reset()
            next_action = self.select_action(self.transformer.transform(observation))
            while not done:
                action = next_action
                next_observation, reward, done, _ = self.env.step(action)

                # transform current and next state
                trans_ob = self.transformer.transform(observation)
                trans_next_ob = None if done else self.transformer.transform(next_observation)                
                next_action = None if done else self.select_action(trans_next_ob)

                td_error = self._td_error(trans_ob, action, reward, trans_next_ob, next_action)
                self._update_estimate(trans_ob, action, td_error)
                observation = copy(next_observation)

        return info

        
    @abstractmethod
    def _td_error(self, state, action, reward, next_state):
        pass

    @abstractmethod
    def _update_estimate(self, state, action, td_error):
        pass
    

