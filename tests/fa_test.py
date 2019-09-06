import unittest
from utils import EnvFactory
import numpy as np
from utils.features_transformer import LinearFeatureTransformer, PolynomialFeatureTransformer, TileFeatureTransformer
import estimators as e 

class Test(unittest.TestCase):
    def setUp(self):
        self.env = EnvFactory.getEnvironment('easy21-v0', 0)
        
    
    def test1(self):
        self.env.reset()
        transf = LinearFeatureTransformer(2,2)
        a, _, _, _ = self.env.step(1)
        at = transf.transform(a,1)
        print(at)
        print(transf.invtransform(at))
        a_, _= transf.invtransform(at)
        self.assertTrue(np.all(a_==a), "Not Equal")
        
    def test2(self):
        self.env.reset()
        transf = PolynomialFeatureTransformer(2,2)
        a, _, _, _ = self.env.step(1)
        at = transf.transform(a,1)
        print(at)
        print(transf.invtransform(at))
        a_, _= transf.invtransform(at)
        self.assertTrue(np.all(a_==a), "Not Equal")
        
    
    def test3(self):
        self.env.reset()
        transf = TileFeatureTransformer((0,21),(0,10),(4,3),1,2)
        a, _, _, _ = self.env.step(1)
        print(transf.tiles)
        print(a)

        at = transf.transform(a,1)
        print(at)
        
    def test4(self):
        a = self.env.reset()
        print(a)
        transf = TileFeatureTransformer((0,21),(0,10),(4,3),1,2)
        state_shape, action_shape = transf.transformed_shape
        at = transf.transform(a, action=1)
        
        lin = e.LinearEstimator(state_shape+action_shape)
        print(at.shape, lin.w.shape)
        print(lin.w)
        print(lin(transf.transform(a,action=0)))
        
