from utils.state_transformer import StateTransformer, abstractmethod

class DummyTransformer(StateTransformer):

    def __init__(self):
       super().__init__()

    def transform(self, obj, save_transformation=True):
        tuple_obj = tuple(obj)
        self.inv_transformation[tuple_obj] = tuple_obj
        return tuple_obj
    

