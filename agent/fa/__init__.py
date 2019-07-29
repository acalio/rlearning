from utils import printProgressBar
from agent import Agent
from copy import copy
from abc import abstractmethod
from collections import deque
import numpy as np

class ApproximationAgent(Agent):

    def __init__(self, env, discount_factor, transformer, policy, estimator, feature_converter):
        Agent.__init__(self,env, discount_factor, transformer, policy)
        self.alpha = alpha
        self.estimator = estimator
        self.feature_covnerter = feature_converter
        self.state


    def learn(self, episodes):
        printProgressBar(0, episodes, prefix = 'Learning:', suffix = 'Complete', length = 50)
        
        # statistics anout the learning process
        info = {
            'rpe': []
        }

        #create refs for efficiency reasons
        transform = self.transformer.transform
        select_action = self.select_action
        step = self.env.step
        get_target = self._get_target
        update_estimate = self._update_estimate


        for e in range(episodes):
            if (e+1)%100 == 0:
                printProgressBar(e+1, episodes, prefix = 'Learning:', suffix = 'Complete', length = 50)
            
            done = False
            observation = self.env.reset()
            

            while not done:
                trans_observation = transform(observation)
                action = select_action(trans_observation)
                next_observation, reward, done, _ = step(action)
                observation = copy(next_observation)
                
                self.q.append((trans_observation, action, reward))

                if len(self.q) > self.nsteps or done:
                    while True:
                        target = get_target() 
                        update_estimate(target)
                        if not (done and self.q):
                            break
            
        return info

    @abstractmethod
    def _get_target(self):
        pass

    @abstractmethod
    def _update_estimate(self, target):
        pass
