"""
change log:
- Version 1: change the out_grads of `backward` function of `ReLU` layer into inputs_grads instead of in_grads
"""

import numpy as np 
from utils.tools import *
from utils.helpers import *
import math

class Layer(object):
    """
    
    """
    def __init__(self, name):
        """Initialization"""
        self.name = name
        self.training = True  # The phrase, if for training then true
        self.trainable = False # Whether there are parameters in this layer that can be trained

    def forward(self, inputs):
        """Forward pass, reture outputs"""
        raise NotImplementedError

    def backward(self, in_grads, inputs):
        """Backward pass, return gradients to inputs"""
        raise NotImplementedError

    def update(self, optimizer):
        """Update parameters in this layer"""
        pass

    def set_mode(self, training):
        """Set the phrase/mode into training (True) or tesing (False)"""
        self.training = training

    def set_trainable(self, trainable):
        """Set the layer can be trainable (True) or not (False)"""
        self.trainable = trainable

    def get_params(self, prefix):
        """Reture parameters and gradients of this layer"""
        return None


class FCLayer(Layer):
    def __init__(self, in_features, out_features, name='fclayer', initializer=Guassian()):
        """Initialization

        # Arguments
            in_features: int, the number of inputs features
            out_features: int, the numbet of required outputs features
            initializer: Initializer class, to initialize weights
        """
        super(FCLayer, self).__init__(name=name)
        self.trainable = True

        self.weights = initializer.initialize((in_features, out_features))
        self.bias = np.zeros(out_features)

        self.w_grad = np.zeros(self.weights.shape)
        self.b_grad = np.zeros(self.bias.shape)

    def forward(self, inputs):
        """Forward pass

        # Arguments
            inputs: numpy array with shape (batch, in_features)

        # Returns
            outputs: numpy array with shape (batch, out_features)
        """
        outputs = None
        bias = list(map(lambda x: self.bias, np.zeros(inputs.shape[0])))

        outputs = np.dot(inputs, self.weights) + np.array(bias)

        return outputs

    def backward(self, in_grads, inputs):
        """Backward pass, store gradients to self.weights into self.w_grad and store gradients to self.bias into self.b_grad

        # Arguments
            in_grads: numpy array with shape (batch, out_features), gradients to outputs
            inputs: numpy array with shape (batch, in_features), same with forward inputs

        # Returns
            out_grads: numpy array with shape (batch, in_features), gradients to inputs
        """
        out_grads = None
        self.w_grad = np.dot(inputs.T, in_grads)

        for i in range(self.bias.shape[0]):
            self.b_grad[i] = np.sum(in_grads[:, i])

        out_grads = np.dot(in_grads, (self.weights).T)
        return out_grads

    def update(self, params):
        """Update parameters (self.weights and self.bias) with new params
        
        # Arguments
            params: dictionary, one key contains 'weights' and the other contains 'bias'

        # Returns
            none
        """
        for k,v in params.items():
            if 'weights' in k:
                self.weights = v
            else:
                self.bias = v
        
    def get_params(self, prefix):
        """Return parameters (self.weights and self.bias) as well as gradients (self.w_grad and self.b_grad)
        
        # Arguments
            prefix: string, to contruct prefix of keys in the dictionary (usually is the layer-ith)

        # Returns
            params: dictionary, store parameters of this layer, one key contains 'weights' and the other contains 'bias'
            grads: dictionary, store gradients of this layer, one key contains 'weights' and the other contains 'bias'

            None: if not trainable
        """
        if self.trainable:
            params = {
                prefix+':'+self.name+'/weights': self.weights,
                prefix+':'+self.name+'/bias': self.bias
            }
            grads = {
                prefix+':'+self.name+'/weights': self.w_grad,
                prefix+':'+self.name+'/bias': self.b_grad
            }
            return params, grads
        else:
            return None

class Convolution(Layer):
    def __init__(self, conv_params, initializer=Guassian(), name='conv'):
        """Initialization

        # Arguments
            conv_params: dictionary, containing these parameters:
                'kernel_h': The height of kernel.
                'kernel_w': The width of kernel.
                'stride': The number of pixels between adjacent receptive fields in the horizontal and vertical directions.
                'pad': The number of pixels padded to the bottom, top, left and right of each feature map. Here, pad=2 means a 2-pixel border of padded with zeros.
                'in_channel': The number of input channels.
                'out_channel': The number of output channels.
            initializer: Initializer class, to initialize weights
        """
        super(Convolution, self).__init__(name=name)
        self.trainable = True
        self.kernel_h = conv_params['kernel_h'] # height of kernel
        self.kernel_w = conv_params['kernel_w'] # width of kernel
        self.pad =  conv_params['pad']
        self.stride = conv_params['stride']
        self.in_channel = conv_params['in_channel']
        self.out_channel = conv_params['out_channel']

        self.weights = initializer.initialize((self.out_channel, self.in_channel, self.kernel_h, self.kernel_w))
        self.bias = np.zeros((self.out_channel))

        self.w_grad = np.zeros(self.weights.shape)
        self.b_grad = np.zeros(self.bias.shape)

    def forward(self, inputs):
        """Forward pass

        # Arguments
            inputs: numpy array with shape (batch, in_channel, in_height, in_width)

        # Returns
            outputs: numpy array with shape (batch, out_channel, out_height, out_width)
        """
        outputs = None
        num_filters, bias, kernel_h, kernel_w, pad, stride = self.out_channel, self.bias, self.kernel_h, self.kernel_w, self.pad, self.stride
        batch_size, in_channel, h_x, w_x = inputs.shape

        h_out = (h_x - kernel_h + 2 * pad) / stride + 1
        w_out = (w_x - kernel_w + 2 * pad) / stride + 1

        h_out, w_out = int(h_out), int(w_out)

        padded_input = process_padding(inputs, pad)
        X_col = np.zeros((batch_size, in_channel*kernel_h*kernel_w, h_out*w_out))

        for i in range(batch_size):
            X = []
            for k in range(h_out):
                for l in range(w_out):
                    x_vec = padded_input[i, 0:padded_input.shape[1], k*stride:k*stride + kernel_h, l*stride:l*stride + kernel_w].reshape(-1)
                    X.append(x_vec)
            X_col[i] = np.asarray(X).T

        X_col_reshaped = np.zeros((in_channel*kernel_h*kernel_w, batch_size*h_out*w_out))
        for i in range(in_channel*kernel_h*kernel_w):
            X_col_reshaped[i, :] = (X_col[:, i, :].T).reshape(batch_size*h_out*w_out)

        self.X_col = X_col_reshaped
        W_col = self.weights.reshape(num_filters, -1)

        bias = transform_bias(bias, X_col_reshaped.shape[1])
        outputs = np.dot(W_col, X_col_reshaped) + bias
        outputs = outputs.reshape(num_filters, h_out, w_out, batch_size)
        outputs = outputs.transpose(3, 0, 1, 2)

        return outputs

    def backward(self, in_grads, inputs):
        """Backward pass, store gradients to self.weights into self.w_grad and store gradients to self.bias into self.b_grad

        # Arguments
            in_grads: numpy array with shape (batch, out_channel, out_height, out_width), gradients to outputs
            inputs: numpy array with shape (batch, in_channel, in_height, in_width), same with forward inputs

        # Returns
            out_grads: numpy array with shape (batch, in_channel, in_height, in_width), gradients to inputs
        """ 
        out_grads = None
        num_filters, bias, field_height, field_width, padding, stride = self.out_channel, self.bias, self.kernel_h, self.kernel_w, self.pad, self.stride

        for i in range(self.out_channel):
            self.b_grad[i] = np.sum(in_grads[:,i,:,:])

        # Reshape from 10,64,15,15 to 64,2250
        in_grads_reshaped = in_grads.transpose(1, 2, 3, 0).reshape(self.out_channel, -1)
        X_col = self.X_col

        # (64, 2250) *(kernel_h * kernel_w, 2250).T
        self.w_grad = np.dot(in_grads_reshaped, X_col.T)
        self.w_grad = self.w_grad.reshape(self.weights.shape)

        W_reshaped = self.weights.reshape(num_filters, -1)
        out_grads = np.dot(W_reshaped.T, in_grads_reshaped)

        N, C, H, W = inputs.shape
        H_padded, W_padded = H + 2 * padding, W + 2 * padding
        x_padded = np.zeros((N, C, H_padded, W_padded), dtype=out_grads.dtype)
        out_height = int((H + 2 * padding - field_height) / stride + 1)
        out_width = int((W + 2 * padding - field_width) / stride + 1)

        i0 = np.repeat(np.arange(field_height), field_width)
        i0 = np.tile(i0, C)
        i1 = stride * np.repeat(np.arange(out_height), out_width)
        j0 = np.tile(np.arange(field_width), field_height * C)
        j1 = stride * np.tile(np.arange(out_width), out_height)
        i_temp = i0.reshape(-1, 1) + i1.reshape(1, -1)
        j_temp = j0.reshape(-1, 1) + j1.reshape(1, -1)

        k_temp = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

        k, i, j = (k_temp.astype(int), i_temp.astype(int), j_temp.astype(int))


        cols_reshaped = out_grads.reshape(C * field_height * field_width, -1, N)
        cols_reshaped = cols_reshaped.transpose(2, 0, 1)
        np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
        if padding == 0:
            out_grads = x_padded
        else:
            out_grads = x_padded[:, :, padding:-padding, padding:-padding]

        return out_grads

    def update(self, params):
        """Update parameters (self.weights and self.bias) with new params
        
        # Arguments
            params: dictionary, one key contains 'weights' and the other contains 'bias'

        # Returns
            none
        """
        for k,v in params.items():
            if 'weights' in k:
                self.weights = v
            else:
                self.bias = v

    def get_params(self, prefix):
        """Return parameters (self.weights and self.bias) as well as gradients (self.w_grad and self.b_grad)
        
        # Arguments
            prefix: string, to contruct prefix of keys in the dictionary (usually is the layer-ith)

        # Returns
            params: dictionary, store parameters of this layer, one key contains 'weights' and the other contains 'bias'
            grads: dictionary, store gradients of this layer, one key contains 'weights' and the other contains 'bias'

            None: if not trainable
        """
        if self.trainable:
            params = {
                prefix+':'+self.name+'/weights': self.weights,
                prefix+':'+self.name+'/bias': self.bias
            }
            grads = {
                prefix+':'+self.name+'/weights': self.w_grad,
                prefix+':'+self.name+'/bias': self.b_grad
            }
            return params, grads
        else:
            return None

class ReLU(Layer):
    def __init__(self, name='relu'):
        """Initialization
        """
        super(ReLU, self).__init__(name=name)

    def forward(self, inputs):
        """Forward pass

        # Arguments
            inputs: numpy array

        # Returns
            outputs: numpy array
        """
        outputs = np.maximum(0, inputs)
        return outputs

    def backward(self, in_grads, inputs):
        """Backward pass

        # Arguments
            in_grads: numpy array, gradients to outputs
            inputs: numpy array, same with forward inputs

        # Returns
            out_grads: numpy array, gradients to inputs 
        """
        inputs_grads = (inputs >=0 ) * in_grads
        out_grads = inputs_grads
        return out_grads


# TODO: add padding
class Pooling(Layer):
    def __init__(self, pool_params, name='pooling'):
        """Initialization

        # Arguments
            pool_params is a dictionary, containing these parameters:
                'pool_type': The type of pooling, 'max' or 'avg'
                'pool_h': The height of pooling kernel.
                'pool_w': The width of pooling kernel.
                'stride': The number of pixels between adjacent receptive fields in the horizontal and vertical directions.
                'pad': The number of pixels that will be used to zero-pad the input in each x-y direction. Here, pad=2 means a 2-pixel border of padding with zeros.
        """
        super(Pooling, self).__init__(name=name)
        self.pool_type = pool_params['pool_type']
        self.pool_height = pool_params['pool_height']
        self.pool_width = pool_params['pool_width']
        self.stride = pool_params['stride']
        self.pad = pool_params['pad']

    def forward(self, inputs):
        """Forward pass

        # Arguments
            inputs: numpy array with shape (batch, in_channel, in_height, in_width)

        # Returns
            outputs: numpy array with shape (batch, in_channel, out_height, out_width)
        """
        outputs = None
        inputs = process_padding(inputs, self.pad)
        out_h = int((inputs.shape[2] + self.pad * 2 - self.pool_height) / self.stride + 1)
        out_w = int((inputs.shape[3] + self.pad * 2 - self.pool_width) / self.stride + 1)
        outputs = np.zeros((inputs.shape[0], inputs.shape[1], out_h, out_w))
        self.loc = np.zeros(outputs.shape, dtype=object)

        for d in range(inputs.shape[0]):
            for c in range(inputs.shape[1]):
                for i in range(out_h):
                    for j in range(out_w):
                        temp = inputs[d, c, i*self.stride:i*self.stride + self.pool_height, j*self.stride:j*self.stride+self.pool_width]
                        if(self.pool_type == 'avg'):
                            outputs[d, c, i, j] = np.average(temp)
                        else:
                            outputs[d, c, i, j] = np.max(temp)

        return outputs
        
    def backward(self, in_grads, inputs):
        """Backward pass

        # Arguments
            in_grads: numpy array with shape (batch, in_channel, out_height, out_width), gradients to outputs
            inputs: numpy array with shape (batch, in_channel, in_height, in_width), same with forward inputs

        # Returns
            out_grads: numpy array with shape (batch, in_channel, in_height, in_width), gradients to inputs
        """
        out_grads = np.zeros(inputs.shape)
        out_h = int((inputs.shape[2] + self.pad * 2 - self.pool_height) / self.stride + 1)
        out_w = int((inputs.shape[3] + self.pad * 2 - self.pool_width) / self.stride + 1)
        if(self.pool_type == 'max'):
            for d in range(in_grads.shape[0]):
                for c in range(in_grads.shape[1]):
                    for i in range(out_h):
                        for j in range(out_w):
                            inputs_portion = inputs[d, c, i*self.stride:i*self.stride+self.pool_height, j*self.stride:j*self.stride+self.pool_width]
                            # ii, jj = self.loc[d, c, i, j]
                            # out_grads[d, c, ii, jj] += in_grads[d, c, i, j]

                            # Find the index of the corresponding max element from inptus
                            temp = np.where(inputs_portion == np.max(inputs_portion))
                            out_grads[d, c, i*self.stride + temp[0][0], j*self.stride+temp[1][0]] += in_grads[d, c, i, j]

        # I have to check self.pool_type outside for-loops because of some weird Python Indentation problems.
        if(self.pool_type == 'avg'):
        	for d in range(in_grads.shape[0]):
        		for c in range(in_grads.shape[1]):
        			for i in range(out_h):
        				for j in range(out_w):
        					out_grads[d, c, i*self.stride:i*self.stride+self.pool_height, j*self.stride:j*self.stride+self.pool_width] += (in_grads[d, c, i, j] / (self.pool_height * self.pool_width))
        return out_grads

class Dropout(Layer):
    def __init__(self, ratio, name='dropout', seed=None):
        """Initialization

        # Arguments
            ratio: float [0, 1], the probability of setting a neuron to zero
            seed: int, random seed to sample from inputs, so as to get mask. (default as None)
        """
        super(Dropout, self).__init__(name=name)
        self.ratio = ratio
        self.mask = None
        self.seed = seed

    def forward(self, inputs):
        """Forward pass (Hint: use self.training to decide the phrase/mode of the model)

        # Arguments
            inputs: numpy array
        # Returns
            outputs: numpy array
        """
        outputs = None

        if(self.seed):
            np.random.seed(self.seed)

        if(self.training):
            self.mask = np.random.binomial(1, self.ratio, size=inputs.shape)
            outputs = self.mask * inputs
            outputs = outputs / (1 - self.ratio) 
        else:
            outputs = inputs
        return outputs

    def backward(self, in_grads, inputs):
        """Backward pass

        # Arguments
            in_grads: numpy array, gradients to outputs
            inputs: numpy array, same with forward inputs

        # Returns
            out_grads: numpy array, gradients to inputs 
        """
        out_grads = None
        out_grads = (in_grads * self.mask) / (1 - self.ratio)
        return out_grads

class Flatten(Layer):
    def __init__(self, name='flatten', seed=None):
        """Initialization
        """
        super(Flatten, self).__init__(name=name)

    def forward(self, inputs):
        """Forward pass

        # Arguments
            inputs: numpy array with shape (batch, in_channel, in_height, in_width)

        # Returns
            outputs: numpy array with shape (batch, in_channel*in_height*in_width)
        """
        batch = inputs.shape[0]
        outputs = inputs.copy().reshape(batch, -1)
        return outputs

    def backward(self, in_grads, inputs):
        """Backward pass

        # Arguments
            in_grads: numpy array with shape (batch, in_channel*in_height*in_width), gradients to outputs
            inputs: numpy array with shape (batch, in_channel, in_height, in_width), same with forward inputs

        # Returns
            out_grads: numpy array with shape (batch, in_channel, in_height, in_width), gradients to inputs 
        """
        out_grads = in_grads.copy().reshape(inputs.shape)
        return out_grads
        
