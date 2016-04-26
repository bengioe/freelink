'''
created on Mar 23, 2016

@author: leolong
'''

import theano
import theano.tensor as T
import numpy
import numpy.linalg
import cPickle
import sys
import random
import time

from collections import OrderedDict


from model import LSTMLayer


config = theano.config
config.compute_test_value = 'off'
_doProfile = False

def sgd(lr):
    def func(params, grads):
        updates = [(i, i - lr * gi) for i, gi in zip(params, grads)]
        return updates

    return func

class Adam:
    def __init__(self, lr = 0.0005,
                 beta1 = 0.9, beta2 = 0.999, epsilon = 1e-4):
        self.lr = lr
        self.b1 = numpy.float32(beta1)
        self.b2 = numpy.float32(beta2)
        self.eps = numpy.float32(epsilon)

    def __call__(self, params, grads):
        t = theano.shared(numpy.array(2., dtype = 'float32'))
        updates = OrderedDict()
        updates[t] = t + 1

        for param, grad in zip(params, grads):
            last_1_moment = theano.shared(numpy.float32(param.get_value() * 0))
            last_2_moment = theano.shared(numpy.float32(param.get_value() * 0))

            new_last_1_moment = T.cast((1. - self.b1) * grad + self.b1 * last_1_moment, 'float32')
            new_last_2_moment = T.cast((1. - self.b2) * grad**2 + self.b2 * last_2_moment, 'float32')

            updates[last_1_moment] = new_last_1_moment
            updates[last_2_moment] = new_last_2_moment
            updates[param] = (param - (self.lr * (new_last_1_moment / (1 - self.b1**t)) /
                                      (T.sqrt(new_last_2_moment / (1 - self.b2**t)) + self.eps)))

        return updates

def ortho_matrix(ndim):
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype(config.floatX)

class MaskedLSTMLayer:
    def __init__(self, ndim):
        W = numpy.concatenate([ortho_matrix(ndim) for i in range(4)], axis = 1)
        U = numpy.concatenate([ortho_matrix(ndim) for i in range(4)], axis = 1)
        b = numpy.zeros(4 * ndim).astype(config.floatX)

        self.params = [theano.shared(i) for i in [W, U, b]]
        self.ndim = ndim

    def apply(self, x, mask):
        W, U, b = self.params
        ndim = self.ndim

        
        def _slice(x, n, dim):
            return x[:, n * dim:(n + 1) * dim]

        def _step(x_t, mask_t, h_t, c_t):
            preact = T.dot(h_t, U) + x_t

            i = T.nnet.sigmoid(_slice(preact, 0, self.ndim))
            f = T.nnet.sigmoid(_slice(preact, 1, self.ndim))
            o = T.nnet.sigmoid(_slice(preact, 2, self.ndim))
            c = T.tanh(_slice(preact, 3, self.ndim))

            c = f * c_t + i * c
            h = o * T.tanh(c)
            
            h = T.switch(mask_t.dimshuffle(0,'x'), h, h_t)

            return h, c

        state_below = T.dot(x, W) + b

        rval, _ = theano.scan(
                _step, [state_below, mask],
                outputs_info = [T.alloc(numpy.float32(0.), x.shape[1], ndim),
                                T.alloc(numpy.float32(0.), x.shape[1], ndim)],
                profile = _doProfile)

        return rval[0]

class HiddenLayer:
    def __init__(self, nin, nout, activation):
        if activation in ['sigmoid', 'tanh', None]:
            k = numpy.sqrt(6. / (nin + nout))
        elif activation == 'relu':
            k = numpy.sqrt(6. / nin)

        self.W = theano.shared(numpy.random.uniform(-k, k, (nin, nout)).astype(config.floatX))
        self.b = theano.shared(numpy.zeros(nout).astype(config.floatX))
        self.params = [self.W, self.b]
        self.activation = {'sigmoid': T.nnet.sigmoid, 'tanh': T.tanh, 'relu': lambda x : T.maximum(0, x), None : lambda x : x}[activation]

    def apply(self, x):
        return self.activation(T.dot(x, self.W) + self.b)

class Predictor:
    def __init__(self, embeddings, embedding_dim = 300, lstm_dim = 128,
                 use_gate = False, gate_activation = None, crs_term = False,
                 optimization_method = None,
                 use_definitions=True):
        #theano.config.compute_test_value = 'raise'
        x = T.imatrix('input')                  # : (seq_len, minibatch_size)
        x.tag.test_value = numpy.int32(numpy.random.randint(0,1000, (50, 32)))
        x_mask = T.imatrix('xmask')              # : (seq_len, minibatch_size) \in {0,1}
        x_mask.tag.test_value = numpy.int32(numpy.random.randint(0,2, (50, 32)))
        y = T.ivector('targets')                # : (minibatch_size)
        y.tag.test_value = numpy.int32(numpy.random.randint(0,2, (32,)))
        if use_definitions:
            defs = T.imatrix('defs')                # : (def_len, minibatch_size)
            defs.tag.test_value = numpy.int32(numpy.random.randint(0,1000,(25, 32)))
            defs_mask = T.imatrix('defmask')        # : (def_len, minibatch_size) \in {0,1}
            defs_mask.tag.test_value = numpy.int32(numpy.random.randint(0,2,(25, 32)))
        else:
            e = T.matrix('e') # (minibatch_size, 300)
            e.tag.test_value = numpy.random.random((32, 300))

        x_embds = embeddings[x.flatten()].reshape(list(x.shape)+[300])
        self.in2lstm = HiddenLayer(embedding_dim, lstm_dim, None)
        self.lstm = MaskedLSTMLayer(lstm_dim)

        self.lstm2pred = HiddenLayer(lstm_dim, 1, None)
        self.e2pred = HiddenLayer(embedding_dim, 1, None)

        self.deflstm = MaskedLSTMLayer(embedding_dim)
        
        if use_gate:
            self.lstm2gate = HiddenLayer(lstm_dim, embedding_dim, None)
            self.e2gate = HiddenLayer(embedding_dim, embedding_dim, None)

        if crs_term:
            self.crs_W = theano.shared(numpy.random.uniform(-.1, .1, (embedding_dim, lstm_dim)).astype(config.floatX), 'crs_W')
            self.crs_b = theano.shared(numpy.float32(0.), 'crs_b')

        prelstm = self.in2lstm.apply(x_embds)
        hs = self.lstm.apply(prelstm, x_mask)               # : (seq_len, minibatch_size, lstm_dim)

        hs = hs[-1] # (minibatch_size, lstm_dim)

        if use_definitions:
            def_embds = embeddings[defs.flatten()].reshape(list(defs.shape)+[300])
            def_encodings = self.deflstm.apply(def_embds, defs_mask)
            def_encodings = def_encodings[-1]
        else:
            def_encodings = e
        

        if use_gate:
            act = {'sigmoid': T.nnet.sigmoid, 'relu': lambda x : T.maximum(0, x), None: lambda x : x}[gate_activation]
            filter_ = act(self.lstm2gate.apply(hs) + self.e2gate.apply(def_encodings))
            classifier_es = filter_ * def_encodings
        else:
            classifier_es = def_encodings

        h_pred = self.lstm2pred.apply(hs)           # : (minibatch_size, 1)
        e_pred = self.e2pred.apply(classifier_es)   # : (minibatch_size, 1)
        h_pred = h_pred.flatten()
        e_pred = e_pred.flatten()

        if crs_term:
            crs = T.sum(T.dot(classifier_es, self.crs_W) * hs, axis=1) + self.crs_b
            pred = T.nnet.sigmoid(h_pred + e_pred + crs)
        else:
            pred = T.nnet.sigmoid(h_pred + e_pred)

        train_cost = T.mean(T.sum(T.nnet.binary_crossentropy(pred, y)))
        train_error = T.sum(T.neq(T.round(pred), y))
        test_error = train_error

        params = [embeddings] + self.in2lstm.params + self.lstm.params + self.lstm2pred.params + self.e2pred.params

        if use_definitions:
            params += self.deflstm.params
            
        if use_gate:
            params += self.lstm2gate.params + self.e2gate.params

        if crs_term:
            params += [self.crs_W, self.crs_b]

        self.params = params
        grads = T.grad(train_cost, params)
        updates = optimization_method(params, grads)
        
        if use_definitions:
            inputs = [x, y, defs, x_mask, defs_mask]
        else:
            inputs = [x, y, e, x_mask]
            
        self.learn = theano.function(inputs, [train_cost, train_error], updates = updates, profile = _doProfile)
        self.test = theano.function(inputs, [test_error])

    def save_params(self, path):
        cPickle.dump([i.get_value() for i in self.params], open(path, 'w'), -1)

    def load_params(self, path):
        values = cPickle.load(open(path, 'r'))
        for v, p in zip(values, self.params):
            p.set_value(v)
