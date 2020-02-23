from abc import ABC, abstractmethod
import torch

class Estimator(ABC):

    def __init__(self, learning_rate):
        self._learning_rate = learning_rate

    @abstractmethod
    def predict(self, observation):
        pass

    @abstractmethod
    def compute_loss(self, target, predicted):
        pass

    @abstractmethod
    def update_weights(self):
        pass

    
    @staticmethod
    def init_weights(shape, strategy):
        tensor = torch.zeros(shape, dtype=torch.float)
        if strategy == 'normal':
            tensor.normal_(0,0.5)
        elif strategy == 'uniform':
            tensor.random_(-1,1)
        tensor = tensor.double()
        tensor.requires_grad = True
        return tensor


class LinearEstimator(Estimator):

    def __init__(self, shape, learning_rate, weight_initialization = 'normal'):
        """
        Shape of the linear estimator. It can either be
        a one or two dimensional vector. 
        One dimension of shape M where N matches
        the number of state-features. 
        Two dimensions with shape N,M where N
        is the number of features and M is the
        number of actions
        """
        super().__init__(learning_rate)
        self.w = Estimator.init_weights(shape, weight_initialization)

        self.y_value, self.loss_value = None, None
        if self.w.ndim>1:
            self.control = True
            _, number_of_actions = self.w.shape
            self.y_fn = lambda x : torch.mm(torch.from_numpy(x.reshape(1,-1)), self.w).squeeze()
            self.loss_fn = lambda target, action_taken : torch.dot(
                                                            torch.sub(self.y_value, target),
                                                            torch.nn.functional.one_hot(torch.tensor(action_taken),number_of_actions).double())
        else:
            self.control = False
            # state-value function
            self.y_fn = lambda x  : torch.dot(x, self.w)
            self.loss_fn = lambda pred, target : torch.sub(pred, target)



    def predict(self, observation):
        self.y_value = self.y_fn(observation)
        return self.y_value

    def compute_loss(self, target, predicted):
        if self.y_value is None:
            raise RuntimeError("You need to call predict first")

        self.loss_value = self.loss_fn(target, predicted)

    def update_weights(self):
        if not self.loss_value:
            raise RuntimeError("You need to call compute_loss first")
        
        #compute the gradient
        self.loss_value.backward()
        
        #update the weights
        self.w -= self.loss_value*self.w.grad 

        # reset loss
        self.loss_value = None

        # reset y_value 
        self.y_value = None

        # reset the gradients
        self.w.grad.zero_()
        
