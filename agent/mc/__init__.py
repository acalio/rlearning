from agent import Agent
from utils import printProgressBar
import numpy as np

class MCAgent(Agent):
    """
    Base class for MonteCarlo based
    algorithms
    """
    
    def __init__(self, env, discount_factor, transformer, policy, every_visit = False):
        super().__init__(env, discount_factor, transformer, policy)
        self._every_visit = every_visit
        
        
    def learn(self, num_of_episodes):
        """

        """
        # create aliases for performace reasons
        gen_ep = self.generate_episode
        is_fist_visit = self._is_first_visit
        every_visit = self._every_visit
        compute_return = self._return
        update = self._update
        new_episode = self._start_episode

        printProgressBar(0, num_of_episodes, prefix = 'Learning:', suffix = 'Complete', length = 50)
        for i in range(num_of_episodes):
            new_episode()
            if (i+1)%100 == 0:
                printProgressBar(i+1, num_of_episodes, prefix = 'Learning:', suffix = 'Complete', length = 50)


            states, actions, rewards = gen_ep(as_separate_array=True)
            
            greturn = 0
            for i in range(len(states)):
                s, a, r = states[-i-1], int(actions[-i-1]), rewards[-i-1]
                greturn = compute_return(s, a, r, greturn)
                #greturn = gamma * greturn + r
                
                if every_visit or is_fist_visit(s, states[:-i-1]):
                    update(greturn, s, a)


    def select_action(self, state):
        state_transformed = self.transformer.transform(state)
        return self.policy(state_transformed)

    def _is_first_visit(self, state, previous_states):
        for s in previous_states:
            if np.all(state==s):
                return False
        return True

    def _return(self, s, action, reward, curr_return):
        raise NotImplementedError

    def _update(self, greturn, state, action):
        raise NotImplementedError

