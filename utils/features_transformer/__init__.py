from utils.state_transformer import StateTransformer
from sklearn.preprocessing import PolynomialFeatureTransformer
import numpy as np
from abc import ABC, abstractmethod

class FeatureTransformer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def transform(self, state, action = None, save_transformation=False):
        pass

    @abstractmethod
    def invtransform(self, state, action = None):
        pass

class LinearFeatureTransformer(FeatureTransformer):
    '''
    It takes a vectorized state-representation as it is
    '''
    def __init__(self, state_shape, action_shape):
        super().__init__()
        self.state_shape, self.action_shape = state_shape, action_shape

    def transform(self, state, action = None, save_transformation=False):
        if action:
            oh_action = np.zeros(self.action_shape)
            oh_action[action] = 1
            return np.concatenate((state, oh_action), axis=0)
        return state        
            
    def invtransform(self, state, action = None):
        if action:
            return state, action
        return state,


class PolynomialFeatureTransformer(FeatureTransformer):
    '''
    It computes a polynomial representation of the
    initial state
    '''    
    def __init__(self, state_dim, degree=2):
        super().__init__()
        self.pol_features = PolynomialFeatureTransformer(degree, include_bias=False)
        self.state_dim = state_dim

    def transform(self, state, action = None, save_transformation=False):
        state = self.pol_features.fit_transform(state) 
        if action:
            oh_action = np.zeros(self.action_shape)
            oh_action[action] = 1
            return np.concatenate((state, oh_action), axis=0)
        return state

    def invtransform(self, state, action = None):
        if action:
            return state[:self.state_dim], action
        return state[:self.state_dim],


class TileFeatureTransformer(StateTransformer):
    '''
    Tile Coding (2-dimensional state)
    '''
    def __init__(self, range_x, range_y, tile, offset):
        '''
        Constructor
        Parameters:
        -----------
        '''
        super().__init__()
        self.tiles = []
        tile_x, tile_y = tile
        start_x, _ = range_x
        start_y, _ = range_y
        for i in np.arange(range_x[0], range_x[1], tile[0]*offset):
            for j in np.arange(range_y[0], range_y[1], tile[1]*offset):
                self.tiles.append((start_x+i,start_y+j,start_x+(i+tile_x),start_y+(j+tile_y)))

    def transform(self, state, action = None, save_transformation=False):
        x1, x2 = state
        state = np.array([1 if blx<x1<=trx and bly < x2 <=try_ else 0 for blx, bly, trx, try_  in self.tiles])
        if action:
            oh_action = np.zeros(self.action_shape)
            oh_action[action] = 1
            return np.concatenate((state, oh_action))
        return state    

    def invtransform(self, state, action = None):
        raise NotImplementedError
        