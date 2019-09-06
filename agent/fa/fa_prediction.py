from agent.fa import ApproximationAgent
from collections import defaultdict
from utils import printProgressBar
import numpy as np
from copy import copy
from collections import deque

class MCPredictionFA(ApproximationAgent):
    
    def __init__(self, env, discount_factor, transformer, policy, estimator, feature_converter):
        super().__init__(env, discount_factor, transformer, policy, estimator, feature_converter)
        self.alpha = 0.00001
        self.statfae_shape = self.env.observation_space.shape
        self.state_visits = defaultdict(int)

    def learn(self, episodes):
        '''
        Implementation of the Gradinete MC algorithm prediction
        algorithm on page 202 of the Sutton and Barto book
        '''
        state_visits = self.state_visits
        printProgressBar(0, episodes, prefix = 'Learning:', suffix = 'Complete', length = 50)
        for i in range(episodes):
            if (i+1)%100 == 0:
                printProgressBar(i+1, episodes, prefix = 'Learning:', suffix = 'Complete', length = 50)

            states, actions, rewards = self.generate_episode(as_separate_array=True,\
                state_dim=self.env.observation_space.shape)

            greturn = 0
            for i in range(len(states)):
                greturn = self.discount_factor*greturn + rewards[-i-1]
             
                state_feature = self.feature_converter.transform(states[-i-1])
                state_transformed = self.transformer.transform(state_feature)
                state_visits[state_transformed] += 1

                error = greturn - self.estimator(state_feature)
                self._update_estimate(error, state_feature)
                # if np.all(np.isnan(self.estimator.w)):
                #     print(error, state_feature, greturn, i)
                #     break

    def select_action(self, state):
        return self.policy(state)
    
    def get_state_value_function(self, **kwargs):
        pass

# class TDPredictionFA(ApproximationAgent):

#     def __init__(self, env, discount_factor, transformer, policy, estimator, feature_converter, alpha):
#         super().__init__(env, discount_factor, transformer, policy, estimator, feature_converter)
#         self.alpha = alpha
#         self.state_visits = defaultdict(int)

#     def learn(self, episodes):
#         '''
#         Implementation of the Semi-gradient TD(0) prediction algorithm
#         on page 203 of the Sutton and Barto book
#         '''
#         transform = self.transformer.transform
#         featurize = self.feature_converter.tranform
#         estimate = self.estimator
#         state_visits = self.state_visits
#         env_reset = self.env.reset
#         env_step = self.env.step
#         select_action = self.select_action
#         state_visits = self.state_visits

#         printProgressBar(0, episodes, prefix = 'Learning:', suffix = 'Complete', length = 50)
#         for e in range(episodes):
#             if (e+1)%100 == 0:
#                 printProgressBar(e+1, episodes, prefix = 'Learning:', suffix = 'Complete', length = 50)
            
#             done = False
#             observation = env_reset()
#             next_action = select_action(transform(featurize(observation)))
#             while not done:
#                 action = next_action
#                 next_observation, reward, done, _ = env_step(action)

#                 # transform current and next state
#                 feature_ob = featurize(observation)
#                 trans_ob = transform(feature_ob)
#                 feature_next_ob = None if done else featurize(next_observation)
#                 trans_next_ob = None if done else transform(feature_next_ob)                
#                 next_action = None if done else select_action(trans_next_ob)
#                 if not feature_next_ob:
#                     error = reward + self.discount_factor*estimate(feature_next_ob) - estimate(feature_ob)
#                 else:
#                     error = reward - estimate(feature_ob)
                
#                 state_visits[trans_ob] += 1
#                 self.alpha = 1 / state_visits[trans_ob]
#                 self._update_estimate(error, feature_ob)    

#                 observation = copy(next_observation)
    
#     def select_action(self, state):
#         return self.policy(state)

# class nStepPredictionFA(ApproximationAgent):

#     def __init__(self, env, discount_factor, transformer, policy, estimator, feature_converter, alpha):
#         super().__init__(env, discount_factor, transformer, policy, estimator, feature_converter)
#         self.alpha = alpha
#         self.nsteps = nsteps
#         self.state_visits = defaultdict(int)
#         self.q = deque(maxlen=self.nsteps+1)
#         self.gammas = np.array([np.power(self.discount_factor, -i) for i in range(self.nsteps+1)])
        

#     def learn(self, episodes):
#         printProgressBar(0, episodes, prefix = 'Learning:', suffix = 'Complete', length = 50)
        
#         # statistics anout the learning process
#         info = {
#             'rpe': []
#         }

#         #create refs for efficiency reasons
#         transform = self.transformer.transform
#         featurize = self.feature_converter.tranform
#         estimate = self.estimator
#         state_visits = self.state_visits
#         env_reset = self.env.reset
#         env_step = self.env.step
#         select_action = self.select_action
#         state_visits = self.state_visits
#         q = self.q

#         get_target = self._get_target

#         for e in range(episodes):
#             if (e+1)%100 == 0:
#                 printProgressBar(e+1, episodes, prefix = 'Learning:', suffix = 'Complete', length = 50)
            
#             done = False
#             observation = env_reset()
            
#             while not done:
#                 observation = featurize(observation)
#                 trans_observation = transform(observation)
#                 action = select_action(trans_observation)

#                 next_observation, reward, done, _ = env_step(action)

#                 self.q.append((observation, action, reward))
#                 observation = copy(next_observation)

#                 if len(self.q) > self.nsteps or done:
#                     while True:
#                         target = get_target() 
#                         s,a,r = q.pop()
#                         s_ = transform(s)
#                         state_visits[s_] += 1
#                         self.alpha = 1/state_visits[s_]
#                         error = target - estimate(s)
#                         self._update_estimate(error,)
#                         if not (done and self.q):
#                             break
            
#         return info

#     def _get_target(self):
#         rewards = [r for _, _ ,r in self.q]
#         rlen = len(rewards)
#         if rlen == self.nsteps:
#             '''
#             it means the agent is not in a terminal state
#             therefore the last reward must be replaced with the q value
#             of the last state action
#             '''
#             last_s, _ , _ = self.q[-1]
#             rewards[-1] = self.estimator[self.feature_converter.transform(last_s)]

#         return np.dot(self.gammas[:rlen], rewards)