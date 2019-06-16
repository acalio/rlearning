from agent import Agent
from utils import EnvFactory, RandomGenerator, plot_V
import numpy as np




SEED = 0
EPISODES = 10

SEED = 0
EPISODES = 500000
DISCOUNT_FACTOR = 1


class MCPredictionAgent(Agent):
    """
    First-Visit MC prediction  
    """

    def __init__(self, env, discount_factor):
        super(self, MCPredictionAgent).__init__(env, discount_factor)
    


def main():
    env = EnvFactory.getEnvironment('easy21-v0', SEED)
    RandomGenerator.get_instance(SEED)

    # initialize Q-table
    low,_ = env.observation_space.low
    high, _ = env.observation_space.high
    state_range = high-low+2

    state_dim = env.observation_space.shape[0]
    V = np.zeros(state_range*state_range).reshape(state_range, state_range)
    state_visits = np.zeros_like(V)

    for _ in range(EPISODES):
        env.reset()
        states, actions, rewards = generate_episode(env, as_separate_array=True, state_dim=state_dim)
        returns = np.flip(np.cumsum(np.flip(rewards)))
        greturn = 0
        for i,state in enumerate(np.flip(states)):
            if i>0 and state in states[:i-1]:
                continue
            greturn = DISCOUNT_FACTOR*greturn + rewards[-i]
            state = tuple(state.astype('int'))
            state_visits[state] += 1
            V[state] += greturn

    V /= np.where(state_visits==0, 1, state_visits)
    plot_V(V, state_range, state_range)

def generate_episode(env, Q, as_separate_array = False, state_dim = None):
    episode = []
    done = False
    while not done:
        action = env.action_space.sample() # random policy
        observation, reward, done, _ = env.step(action)
        episode.append((observation, action, reward))
    
    if as_separate_array:
        states, actions, rewards = np.zeros(len(episode)*state_dim).reshape(-1, state_dim), np.zeros(len(episode)), np.zeros(len(episode))
        for i,(o,a,r) in enumerate(episode):
            states[i], actions[i], rewards[i] = o,a,r
        return states, actions, rewards
    return episode

def eps_soft_selection(Q, state, actions, eps):
    max_action = np.argmax(Q[state])
    return np.random.choice(actions, p=np.where(actions==max_action, 1.0-eps+eps/len(actions), eps/len(actions)))

if __name__ == "__main__":
    main()