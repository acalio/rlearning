from abc import ABC, abstractmethod
import torch

class Estimator(ABC):

    def __init__(self, learning_rate):
        self._learning_rate = learning_rate
        pass

    @abstractmethod
    def predict(self, observation):
        pass

    @abstractmethod
    def compute_loss(self, predicted, target):
        pass

    @abstractmethod
    def update_weights(self):
        pass

    
    @staticmethod
    def init_weights(shape, strategy):
        tensor = torch.zeros(shape)
        if strategy == 'normal':
            tensor.normal_(0,0.5)
        elif strategy == 'uniform':
            tensor.random_(-1,1)
        tensor.requires_grad = True
        return tensor


class LinearEstimator(Estimator):

    def __init__(self, shape, learning_rate, weight_initialization = 'normal'):
        super().__init__(learning_rate)
        self.w = Estimator.init_weights(shape, weight_initialization)
        if self.w.ndim>1:
            self.out = lambda x : torch.mm(torch.from_numpy(x.reshape(-1,1)), self.w)
        else:
            self.out = lambda x  : torch.dot(x, self.w)
        
        self.loss = lambda pred, target : torch.sub(pred, target)
        self.loss_value = None
        
    def predict(self, observation):
        return self.out(observation)

    def compute_loss(self, predicted, target):
        self.loss_value = self.loss(predicted, target)

    def update_weights(self):
        if not self.loss_value:
            raise RuntimeError("You need to call compute_loss first")
        # reset the gradients
        self.w.zero_grad()
        
        #compute the gradient
        self.loss_value.backward()
        
        #update the weights
        self.w = torch.sub(self.w, self._learning_rate*self.w.grad)

        # reset loss
        self.loss_value = None
