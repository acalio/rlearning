from abc import ABC, abstractmethod
import torch

class Estimator(ABC):

    def __init__(self, learning_rate):
        self._learning_rate = learning_rate

    @abstractmethod
    def predict(self, state):
        pass

    @abstractmethod
    def compute_loss(self, target, state, action=None):
        pass


    @abstractmethod
    def update_weights(self, target, state, action=None):
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

    def __init__(self, shape, learning_rate, weight_initialization='uniform'):
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
        self.loss_value = None
        if self.w.ndim>1:
            _, number_of_actions = self.w.shape
            self.y_fn = lambda x: torch.mm(
                torch.from_numpy(x.reshape(1, -1)).double(),
                self.w
            ).squeeze()
            self.loss_fn = lambda target, predicted, action_taken: \
                torch.dot(
                    torch.sub(predicted, target),
                    torch.nn.functional.one_hot(torch.tensor(action_taken), number_of_actions).double()
                )
        else:
            # state-value function
            self.y_fn = lambda x: torch.dot(x, self.w)
            self.loss_fn = lambda pred, target: torch.sub(pred, target)

    def predict(self, state):
        return self.y_fn(state).detach().numpy()

    def __predict(self, state):
        return self.y_fn(state)


    def compute_loss(self, target, state, action=None):
        predicted = self.__predict(state)
        if action is None:
            return self.loss_fn(target, predicted)
        else:
            return self.loss_fn(target, predicted, action)

    def update_weights(self, target, state, action=None):
        #zero the gradient
        if self.w.grad is not None:
            self.w.grad.detach_()
            self.w.grad.zero_()

        #compute the value of the loss function
        self.loss_value = self.compute_loss(target, state, action)

        #compute the gradient
        self.loss_value.backward()

        #update the weights
        self.w.data.sub_(self._learning_rate*self.loss_value*self.w.grad.data)

#        print(self.w.grad)

#        print(self.w.grad)


