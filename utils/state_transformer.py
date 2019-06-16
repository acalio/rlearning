from abc import abstractmethod, ABC

class StateTransformer(ABC):

    def __init__(self):
        self.inv_transformation = {}

    @abstractmethod
    def transform(self, obj, save_transformation=True):
        pass
        
    def invtransform(self, digest):
        return self.inv_transformation[digest]




