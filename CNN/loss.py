import numpy as np
from functools import reduce
import math

class Loss(object):
    
    def __init__(self):
        self.trainable = False # Whether there are parameters in this layer that can be trained
        self.training = False # The phrase, if for training then true

    def forward(self, inputs, targets):
        """Forward pass, reture outputs"""
        raise NotImplementedError

    def backward(self, inputs, targets):
        """Backward pass, return gradients to inputs"""
        raise NotImplementedError

    def set_mode(self, training):
        """Set the phrase/mode into training (True) or tesing (False)"""
        self.training = training


class SoftmaxCrossEntropy(Loss):
    def __init__(self, num_class):
        """Initialization

        # Arguments
            num_class: int, the number of category
        """
        super(SoftmaxCrossEntropy, self).__init__()
        self.num_class = num_class

    def forward(self, inputs, targets):
        """Forward pass

        # Arguments
            inputs: numpy array with shape (batch, num_class)
            targets: numpy array with shape (batch,)

        # Returns
            outputs: float, batch loss
            probs: numpy array with shape (batch, num_class), probability to each category with respect to each image
        """
        outputs = 0
        # This is to clone the inputs
        probs = np.array(list(inputs))

        batch_size, num_class = inputs.shape
        for i in range(batch_size):
            logits_sum = reduce( (lambda x, y: x + math.exp(y)), np.insert(inputs[i, :], 0, 0) )
            probs[i, :] = np.array( list( map( lambda x: (math.exp(x) / logits_sum), inputs[i, :] ) ) )
            # predicted_class = np.argmax(softmax_outputs[i, :])
            # probs[i, predicted_class] = 1
            outputs += (-math.log(probs[i, targets[i]]))

        outputs = outputs / batch_size 
        return outputs, probs

    def backward(self, inputs, targets):
        """Backward pass

        # Arguments
            inputs: numpy array with shape (batch, num_class), same with forward inputs
            targets: numpy array with shape (batch,), same eith forward targets

        # Returns
            out_grads: numpy array with shape (batch, num_class), gradients to inputs 
        """
        out_grads = np.zeros(inputs.shape)
        _, probs = self.forward(inputs, targets)
        batch_size, num_class = inputs.shape
        for i in range(batch_size):
            for j in range(num_class):
                if(j == targets[i]):
                    out_grads[i, j] = probs[i, j] - 1
                else: 
                    out_grads[i, j] = probs[i, j]

        out_grads /= batch_size
        return out_grads

class L2(Loss):
    def __init__(self, w=0.01):
        """Initialization

        # Arguments
            w: float, weight decay ratio.
        """
        self.w = w

    def forward(self, params):
        """Forward pass

        # Arguments
            params: dictionary, store all weights of the whole model

        # Returns
            outputs: float, L2 regularization loss
        """
        loss = 0
        for _, v in params.items():
            loss += np.sum(v**2)
        outputs = 0.5 * self.w * loss
        return outputs

    def backward(self, params):
        """Backward pass

        # Arguments
            params: dictionary, store all weights of the whole model

        # Returns
            out_grads: dictionary, gradients to each weights in params 
        """
        out_grads = {}
        for k, v in params.items():
            out_grads[k] = self.w * params[k]
        return out_grads