from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from abc import ABC, abstractmethod

class FeatureTransformer(ABC):
    def __init__(self, state_shape, action_shape):
        self.state_shape, self.action_shape = state_shape, action_shape

    @abstractmethod
    def transform(self, state, action = None, save_transformation=False):
        pass

    @abstractmethod
    def invtransform(self, state):
        pass

    def _action_encoded(self, state):
        return not np.array_equiv(np.zeros(self.state_shape), np.zeros_like(state))

    def _add_action(self, state, action):
        oh_action = np.zeros(self.action_shape)
        oh_action[action] = 1
        return np.concatenate((state, oh_action), axis=0)

    @property
    @abstractmethod
    def transformed_shape(self):
        pass

class LinearFeatureTransformer(FeatureTransformer):
    '''
    It takes a vectorized state-representation as it is
    '''
    def __init__(self, state_shape, action_shape):
        super().__init__(state_shape, action_shape)

    def transform(self, state, action = None, save_transformation=False):
        if action is not None:
            return self._add_action(state, action)
        return state        
            
    def invtransform(self, state = None):
        if self._action_encoded(state):
            return state[:self.state_shape], np.argmax(state[-self.action_shape:])
        return state,

    def transformed_shape(self):
        return self.state_shape, self.action_shape

class PolynomialFeatureTransformer(FeatureTransformer):
    '''
    It computes a polynomial representation of the
    initial state
    '''    
    def __init__(self, state_shape, action_shape, degree=2):
        super().__init__(state_shape, action_shape)
        self.pol_features = PolynomialFeatures(degree, include_bias=False)
                
    def transform(self, state, action = None, save_transformation=False):
        state = self.pol_features.fit_transform(state.reshape(1,-1)).reshape(-1)
        if action is not None:
            return self._add_action(state, action)
        return state

    def invtransform(self, state):
        if self._action_encoded(state):
            return state[:self.state_shape], np.argmax(state[-self.action_shape:])
        return state[:self.state_shape],

    @property
    def transformed_shape(self):
        return self.pol_features.fit_transform(np.zeros((1,self.state_shape))).reshape(-1).shape, self.action_shape

class TileFeatureTransformer(FeatureTransformer):
    '''
    Tile Coding (2-dimensional state)
    '''
    def __init__(self, range_x, range_y, tile, offset, action_shape):
        '''
        Constructor
        Parameters:
        -----------
        '''
        self.tiles = []
        tile_x, tile_y = tile
        start_x, _ = range_x
        start_y, _ = range_y
        for i in np.arange(range_x[0], range_x[1], tile[0]*offset):
            for j in np.arange(range_y[0], range_y[1], tile[1]*offset):
                self.tiles.append((start_x+i,start_y+j,start_x+(i+tile_x),start_y+(j+tile_y)))
        
        super().__init__(len(self.tiles), action_shape) 

    def transform(self, state, action = None, save_transformation=False):
        x1, x2 = state
        state = np.array([1 if blx<x1<=trx and bly < x2 <=try_ else 0 for blx, bly, trx, try_  in self.tiles])
        if action is not None:
            return self._add_action(state, action)
        return state    

    def invtransform(self, state):
        pass

    @property
    def transformed_shape(self):
        return len(self.tiles), self.action_shape


    