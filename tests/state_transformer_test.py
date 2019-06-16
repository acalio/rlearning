import unittest
from utils import EnvFactory
from agent.mc.mc_control import MCControlAgent
from utils.hash_transformer import HashTransformer
from utils.plots import plot_V, plot_learning_curve
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle 

class Test(unittest.TestCase):
    def setUp(self):
       self.transf = HashTransformer()

    def test1(self):
        a = np.array([15,5], dtype=int)
        d = self.transf.transform(a)
        self.assertTrue(np.all(a==self.transf.invtransform(d)))
        print(self.transf.transform(np.array([15,5])))
        
        
        
