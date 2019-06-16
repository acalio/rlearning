from utils.state_transformer import StateTransformer
import xxhash

class HashTransformer(StateTransformer):

    def __init__(self, bit = 32):
        super().__init__()
        if bit == 32:
            self.h = xxhash.xxh32()
        else:
            self.h = xxhash.xxh64()


    def transform(self, obj, save_transformation=True):
        self.h.reset()
        self.h.update(obj)
        digest = self.h.hexdigest()
        if save_transformation:
            self.inv_transformation[digest] = obj
        return digest