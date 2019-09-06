from utils import printProgressBar
from agent import Agent
from copy import copy
from abc import abstractmethod
from collections import deque
import numpy as np

class nStepAgent(Agent):

    def __init__(self, env, discount_factor, transformer, policy, alpha, nsteps):
        Agent.__init__(self,env, discount_factor, transformer, policy)
        self.alpha = alpha
        self.nsteps = nsteps
        self.q = deque(maxlen=self.nsteps+1)
        self.gammas = np.array([np.power(self.discount_factor, -i) for i in range(self.nsteps+1)])
        

    def learn(self, episodes):
        printProgressBar(0, episodes, prefix = 'Learning:', suffix = 'Complete', length = 50)
        
        # statistics anout the learning process
        info = {
            'rpe': []
        }

        #create refs for efficiency reasons
        select_action = self.select_action
        step = self.env.step
        get_target = self._get_target
        update_estimate = self._update


        for e in range(episodes):
            if (e+1)%100 == 0:
                printProgressBar(e+1, episodes, prefix = 'Learning:', suffix = 'Complete', length = 50)
            
            done = False
            observation = self.env.reset()
            

            while not done:
                
                action = select_action(observation)
                next_observation, reward, done, _ = step(action)
                observation = copy(next_observation)
                
                self.q.append((observation, action, reward))

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
    def _update(self, target):
        pass
    