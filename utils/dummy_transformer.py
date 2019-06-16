from utils.state_transformer import StateTransformer, abstractmethod

class DummyTransformer(StateTransformer):

    def __init__(self):
        super(self, DummyTransformer).__init__()

    def transform(self, obj, save_transformation=True):
        return obj
    

