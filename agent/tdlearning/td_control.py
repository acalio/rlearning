from agent import ControlAgent
from agent.tdlearning import TDAgent
from collections import defaultdict
import numpy as np



class SARSA(TDAgent,ControlAgent):
    """
    On-Policy TD-Control 
    """

    def __init__(self, env, discount_factor, transformer, policy, alpha):
        """Constructor
        
        Parameters
        ----------
            env : openAI environment
            disocunt_factor : float
            transformer : StateTransformer
            policy : Policy, to evaluate
            alpha : float, step size of the update rule
        """
        ControlAgent.__init__(self, env, discount_factor, transformer, policy)
        TDAgent.__init__(self, env, discount_factor, transformer, policy, alpha)
        
        cz = lambda : np.zeros(self.env.action_space.n)
        self.state_action_visit = defaultdict(cz)

    
    def _td_error(self, s, a, r, s_prime, a_prime):
        """
        Compute the TD-Error
        Parameters:
        -----------
            s : current_state (already transformed)
            a : current_action
            r : reward due to action a in state s
            s_prime : state next to s (already transformed)
            a_prime : action next_to
        """
        if s_prime is None and a_prime is None:
            return r - self.Q[s][a]
        return r + self.discount_factor * self.Q[s_prime][a_prime] - self.Q[s][a]


    def _update_estimate(self, state, action, td_error):
        """
        Update the estimate

        Parameters
        ----------
            state : current_state (already transformed)
            action : current action
            td_error: td error
        """
        self.state_action_visit[state][action] += 1
        self.alpha = 1 / self.state_action_visit[state][action]
        self.Q[state][action] += self.alpha * td_error


    def select_action(self, state):
        return self.policy(state, self.Q[state])




class ExpSARSA(SARSA):
    """ Expected SARSA """
    def __init__(self, env, discount_factor, transformer, policy, alpha):
        SARSA.__init__(self, env, discount_factor, transformer, policy, alpha)
        
    def _td_error(self, s, a, r, s_prime, a_prime):
        """
        Compute the TD-Error
        Parameters:
        -----------
            s : current_state (already transformed)
            a : current_action
            r : reward due to action a in state s
            s_prime : state next to s (already transformed)
            a_prime : action next_to
        """
        if s_prime is None:
            exp_next_q = 0
        else:
            exp_next_q = np.dot(
                self.policy.get_actions_probabilities(s_prime, self.Q[s_prime]), # probabilities
                self.Q[s_prime])                                # actions q-values
        
        return r + self.discount_factor * exp_next_q - self.Q[s][a]

    
        
class QLearning(ControlAgent, TDAgent):
    """ Off-Policy TD control """

    def __init__(self, env, discount_factor, transformer, policy, alpha):
        ControlAgent.__init__(self, env, discount_factor, transformer, policy)
        TDAgent.__init__(self, env, discount_factor, transformer, policy, alpha)
        cz = lambda : np.zeros(self.env.action_space.n)
        self.state_action_visit = defaultdict(cz)

    def _td_error(self, s, a, r, s_prime, a_prime):
        """
        Compute the TD-Error
        Parameters:
        -----------
            s : current_state (already transformed)
            a : current_action
            r : reward due to action a in state s
            s_prime : state next to s (already transformed)
            a_prime : action next_to
        """
        if s_prime is None:
            next_max_q = 0
        else:
            next_max_q = np.max(self.Q[s_prime])
        return r + self.discount_factor * next_max_q - self.Q[s][a]
    

    def _update_estimate(self, state, action, td_error):
        self.state_action_visit[state][action] += 1
        self.alpha = 1/self.state_action_visit[state][action]
        self.Q[state][action] += self.alpha * td_error


    def select_action(self, state):
        return self.policy(state, self.Q[state])


class DoubleQLearning(QLearning):
    """ 
    Double Q-Learning algorithm, 
    page 121 Sutton and Barto book
    """
    def __init__(self, env, discount_factor, transformer, policy, alpha):
        ControlAgent.__init__(self, env, discount_factor, transformer, policy)
        QLearning.__init__(self, env, discount_factor, transformer, policy, alpha)
        cz = lambda : np.zeros(self.env.action_space.n)
        self.Q_ = defaultdict(cz)

    def _td_error(self, s, a, r, s_prime, a_prime):
        """
        Compute the TD-Error wrt to each Q-Table
        and return a list of td-errors, with two
        values

        Parameters    
        -----------
            s : current_state (already transformed)
            a : current_action
            r : reward due to action a in state s
            s_prime : state next to s (already transformed)
            a_prime : action next_to
        """
        if s_prime is None:
            next_q_value, next_q_value_ = 0,0
        else:
            next_q_value = self.Q_[s_prime][np.argmax(self.Q[s_prime])]
            next_q_value_ = self.Q[s_prime][np.argmax(self.Q_[s_prime])]
        return [
                r + self.discount_factor * next_q_value - self.Q[s][a], 
                r + self.discount_factor * next_q_value_ - self.Q_[s][a]
            ] 

    def _update_estimate(self, state, action, td_error):
        self.state_action_visit[state][action] += 1
        self.alpha = 1 / self.state_action_visit[state][action]
        if np.random.uniform() <= 0.5:
            self.Q[state][action] += self.alpha * td_error[0]
        else:
            self.Q_[state][action] += self.alpha * td_error[1]

    def select_action(self, state):
        return self.policy(state, self.Q[state] + self.Q_[state])

