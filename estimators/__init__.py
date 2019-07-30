from abc import ABC, abstractmethod
import numpy as np
class Estimator(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, observation):
        pass

    @abstractmethod
    def get_derivative(self):
        pass

    @staticmethod
    def init_weights(shape, strategy):
        if strategy == 'zero':
            return np.zeros(shape)
        elif strategy == 'normal':
            return np.random.normal(0,0.2, shape)
        elif strategy == 'uniform':
            return  np.random.uniform(size=shape)
        elif strategy == 'ones':
            return np.ones(shape)


class LinearEstimator(Estimator):

    def __init__(self, shape, weight_initialization = 'normal'):
        super().__init__()
        self.w = Estimator.init_weights(shape, weight_initialization)

    def __call__(self, observation):
        return np.dot(self.w, observation)

    def get_derivative(self):
        def fprime(observation):
            return observation
        return fprime