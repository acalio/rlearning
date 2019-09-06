from utils.plots import plot_V, plot_learning_curve
import numpy as np
from numpy.linalg import norm
import time
import pickle
import os

RESULT_PATH = "./results"

def isControlAgent(agent):
    for base in agent.__class__.__bases__:
        if base.__name__.lower() == 'controlagent':
            return True
    return False

def string_params(kwargs):
    return "{}".format("_".join(["{}{}".format(k,v)  for k,v in kwargs.items()]))
    


def learnQV(agent, episodes, env, algo, algo_kws = None, save_V = True, save_Q = True):
    player, dealer = env.observation_space.high
    start = time.time()    
    info = agent.learn(episodes)
    print("Learning time: %s s" % (time.time() - start))
    V_matrix = extract_V(agent, env)
    
    algo = "_".join([algo,string_params(algo_kws)])
    plot_V(V_matrix, dealer, player, save=os.path.join(RESULT_PATH, 'img', "{}.pdf".format(algo)))
    if save_V:
        with open(os.path.join(RESULT_PATH,'V{}_{}.pickle'.format(algo, episodes)), 'wb') as f:
                pickle.dump(V_matrix, f)
    
    if isControlAgent(agent) and save_Q:
        Q_matrix = extract_Q(agent, env)
        with open(os.path.join(RESULT_PATH,'Q{}_{}.pickle'.format(algo,episodes)), 'wb') as f:
            pickle.dump(Q_matrix, f)
    

def extract_V(agent, env):
    player, dealer = env.observation_space.high
    transf = agent.transformer
    V = agent.get_state_value_function(optimality=True)
    V_matrix = np.zeros((player, dealer))
    for k,v in V.items():
        state = transf.invtransform(k) - 1
        V_matrix[tuple(state.astype(int))] = v
    
    return V_matrix

def extract_Q(agent, env):
    player, dealer = env.observation_space.high
    transf = agent.transformer
    Q_matrix = np.zeros((player, dealer, env.action_space.n))
    for k,v in agent.Q.items():
        state = tuple(agent.transformer.invtransform(k).astype(int) -1)
        Q_matrix[state] = v
    return Q_matrix
    
def play(agent, env, runs):
    tests = runs
    wl = [0, 0]
    for _ in range(tests):
        done = False
        observation = env.reset()
        reward = 0
        while not done:
            action = agent.select_action(observation)
            observation, reward, done, _ = env.step(action)

        ix = 0 if reward == 1 else 1
        wl[ix] += 1
    print("Percentage of win: {}".format(wl[0]/tests))


def RMSE(agent, episodes, env, algo, target_Q):
    y = []
    num_states, num_actions = len(target_Q), len(next(iter(target_Q)))
    for i in range(episodes):
        agent.learn(1)
        rmse = 0
        for k,v in target_Q.items():
            agent_qs = agent.Q[k]
            rmse += norm(v-agent_qs, ord=2)
        y.append(rmse/ (num_states * num_actions))
    return y

        


