from agent import ControlAgent, ApproximationAgent
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
        TDAgent.__init__(self, env, discount_factor, transformer, policy, alpha)
        ControlAgent.__init__(self, env, discount_factor, transformer, policy)
        
        cz = lambda : np.zeros(self.env.action_space.n)
        self.state_action_visit = defaultdict(cz)

    
    def _td_error(self, state, action, reward, next_state, next_action):
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
        if next_state is not None:
            next_state = self.transformer.transform(next_state)
            next_state_value = self._Q[next_state][next_action]
        else:
            next_state_value = 0

        state = self.transformer.transform(state)
        current_state_value = self._Q[state][action]

        return reward + self.discount_factor * next_state_value - current_state_value


    def _update(self, state, action, td_error):
        """
        Update the estimate

        Parameters
        ----------
            state : current_state (already transformed)
            action : current action
            td_error: td error
        """
        state = self.transformer.transform(state)
        self.state_action_visit[state][action] += 1
        self.alpha = 1 / self.state_action_visit[state][action]
        self._Q[state][action] += self.alpha * td_error


    def select_action(self, state):
        state_transformed = self.transformer.transform(state)
        return self.policy(state, self._Q[state_transformed])



class SARSAFA(SARSA, ApproximationAgent):
    def __init__(self, env, discount_factor, transformer, policy, estimator, feature_converter, learning_rate, every_visit=False):

    def __init__(self, env, discount_factor, transformer, policy, alpha, ):
        pass
    


class ExpSARSA(SARSA):
    """ Expected SARSA """
    def __init__(self, env, discount_factor, transformer, policy, alpha):
        SARSA.__init__(self, env, discount_factor, transformer, policy, alpha)
        
    def _td_error(self, state, action, reward, next_state, next_action):
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
        if next_state is None:
            exp_next_q = 0
        else:
            exp_next_q = np.dot(
                self.policy.get_actions_probabilities(next_state, self.Q[next_state]), # probabilities
                self.Q[next_state])                                # actions q-values
        
        return reward + self.discount_factor * exp_next_q - self.Q[state][action]

    
        
class QLearning(ControlAgent, TDAgent):
    """ Off-Policy TD control """

    def __init__(self, env, discount_factor, transformer, policy, alpha):
        ControlAgent.__init__(self, env, discount_factor, transformer, policy)
        TDAgent.__init__(self, env, discount_factor, transformer, policy, alpha)
        cz = lambda : np.zeros(self.env.action_space.n)
        self.state_action_visit = defaultdict(cz)

    def _td_error(self, state, action, reward, next_state, next_action):
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
        if next_state is None:
            next_max_q = 0
        else:
            next_max_q = np.max(self.Q[next_state])
        return reward + self.discount_factor * next_max_q - self.Q[state][action]
    

    def _update(self, state, action, td_error):
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

    def _td_error(self, state, action, reward, next_state, next_action):
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
        if next_state is None:
            next_q_value, next_q_value_ = 0,0
        else:
            next_q_value = self.Q_[next_state][np.argmax(self.Q[next_state])]
            next_q_value_ = self.Q[next_state][np.argmax(self.Q_[next_state])]
        return [
                reward + self.discount_factor * next_q_value - self.Q[state][action], 
                reward + self.discount_factor * next_q_value_ - self.Q_[state][action]
            ] 

    def _update(self, state, action, td_error):
        self.state_action_visit[state][action] += 1
        self.alpha = 1 / self.state_action_visit[state][action]
        if np.random.uniform() <= 0.5:
            self.Q[state][action] += self.alpha * td_error[0]
        else:
            self.Q_[state][action] += self.alpha * td_error[1]

    def select_action(self, state):
        return self.policy(state, self.Q[state] + self.Q_[state])

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
        for k, action_array in self.Q.items():
            if optimality:
                V[k] = max(np.max(action_array), np.max(self.Q_[k]))
            else:
                max_action = np.argmax(action_array)
                probs = self.policy.get_actions_probabilities(k, action_array + self.Q_[k])
                mean = np.dot(action_array, probs)
                V[k] = mean
        return V