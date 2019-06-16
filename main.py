#!/home/antonio/App/anaconda3/bin/python
from utils import EnvFactory, RandomGenerator, plot_V
from utils.hash_transformer import HashTransformer
import numpy as np
from agent.mc.mc_prediction import MCPredictionAgent

SEED = 0
EPISODES = 10

SEED = 0
EPISODES = 500000
DISCOUNT_FACTOR = 1
EPS = 0.1

def main():
    env = EnvFactory.getEnvironment('easy21-v0', SEED)
    RandomGenerator.get_instance(SEED)
    transf = HashTransformer()
    agent =  MCPredictionAgent(env, 1.0, transf)



if __name__ == "__main__":
    main()