import gym
import easy21_env
import numpy as np

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()


class EnvFactory:
    @staticmethod
    def getEnvironment(env_name, seed = None):
        env = gym.make(env_name)
        env.seed(seed)
        return env 

class RandomGenerator:
    class __RandomGenerator:
        def __init__(self, seed = None):
            np.random.seed(seed)


        def get_next(self, _min, _max):
            return np.random.randint(_min, _max)

    __instance = None

    def __init__(self, seed = None):
        if RandomGenerator.__instance is None:
            RandomGenerator.__instance = RandomGenerator.__RandomGenerator(seed)


    def get_next(self, _min, _max):
        return RandomGenerator.__instance.get_next(_min, _max)


    @staticmethod
    def get_instance(seed=None):
        return RandomGenerator(seed)

