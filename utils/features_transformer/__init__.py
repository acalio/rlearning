from utils.state_transformer import StateTransformer
from sklearn.preprocessing import PolynomialFeatureTransformer
class LinearFeatureTransformer(StateTransformer):
    '''
    It takes a vectorized state-representation as it is
    '''
    def __init__(self):
        super().__init__()


    def transform(self, state, save_transformation=False):
        return state

    def invtransform(self, state):
        return state


class PolynomialFeatureTransformer(StateTransformer):
    '''
    It computes a polynomial representation of the
    initial state
    '''    
    def __init__(self, state_dim, degree=2):
        super().__init__()
        self.pol_features = PolynomialFeatureTransformer(degree, include_bias=False)
        self.state_dim = state_dim

    def transform(self, obj, save_transformation=False):
        return self.pol_features.fit_transform(obj)

    def invtransform(self, state):
        return state[:self.state_dim]


class TileFeatureTransformer(StateTransformer):
    '''
    Tile Coding
    '''
    def __init__(self, state_dim, degree=2):
        super().__init__()
        self.pol_features = PolynomialFeatureTransformer(degree, include_bias=False)
        self.state_dim = state_dim

    def transform(self, obj, save_transformation=False):
        return self.pol_features.fit_transform(obj)

    def invtransform(self, state):
        return state[:self.state_dim]