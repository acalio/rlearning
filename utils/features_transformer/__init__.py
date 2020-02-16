from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from abc import ABC, abstractmethod
'''
Abstract class for every
feature transform. Every
transformer takes as input
a state of the environment
and it return its feature-representation
'''
class FeatureTransformer(ABC):
    def __init__(self, shape):
        self._shape = shape

    @abstractmethod
    def transform(self, state, save_transformation=False):
        pass

    @abstractmethod
    def invtransform(self, state):
        pass
    
    @property
    def tranformed_shape(self):
        pass

class LinearFeatureTransformer(FeatureTransformer):
    '''
    It takes a vectorized state-representation as it is
    '''
    def __init__(self, state_shape):
        super().__init__(state_shape)

    def transform(self, state, save_transformation=False):
        return state        
            
    def invtransform(self, state = None):
        return state,

    @property
    def tranformed_shape(self):
        return self._shape

class PolynomialFeatureTransformer(FeatureTransformer):
    '''
    It computes a polynomial representation of the
    initial state
    '''    
    def __init__(self, state_shape, degree=2):
        super().__init__(state_shape)
        self.pol_features = PolynomialFeatures(degree, include_bias=False)
                
    def transform(self, state, save_transformation=False):
        state = self.pol_features.fit_transform(state.reshape(1,-1)).ravel()
        return state

    @property
    def invtransform(self, state):
        pass

    
    def tranformed_shape(self):
        return self.pol_features.fit_transform(np.zeros(self._shape).reshape(1,-1)).ravel().shape

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

    def transform(self, state, save_transformation=False):
        x1, x2 = state
        state = np.array([1 if blx<x1<=trx and bly < x2 <=try_ else 0 for blx, bly, trx, try_  in self.tiles])
        return state    

    def invtransform(self, state):
        pass




    