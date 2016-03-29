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

class LSTMLayer:
    def __init__(self, ndim):
        W = numpy.concatenate([ortho_matrix(ndim) for i in range(4)], axis = 1)
        U = numpy.concatenate([ortho_matrix(ndim) for i in range(4)], axis = 1)
        b = numpy.zeros(4 * ndim).astype(config.floatX)

        self.params = [theano.shared(i) for i in [W, U, b]]
        self.ndim = ndim

    def apply(self, x):
        W, U, b = self.params
        ndim = self.ndim

        def _slice(x, n, dim):
            return x[:, n * dim:(n + 1) * dim]

        def _step(x_t, h_t, c_t):
            preact = T.dot(h_t, U) + x_t

            i = T.nnet.sigmoid(_slice(preact, 0, self.ndim))
            f = T.nnet.sigmoid(_slice(preact, 1, self.ndim))
            o = T.nnet.sigmoid(_slice(preact, 2, self.ndim))
            c = T.tanh(_slice(preact, 3, self.ndim))

            c = f * c_t + i * c
            h = o * T.tanh(c)

            return h, c

        state_below = T.dot(x, W) + b

        rval, _ = theano.scan(
                _step, [state_below],
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
    def __init__(self, embeddings, embedding_dim = 300, lstm_dim = 128, use_gate = False, gate_activation = None, optimization_method = None):
        x = T.imatrix('input')              # : (seq_len, minibatch_size)
        y = T.tensor3('targets')            # : (nblanks, minibatch_size, 2)
        e = T.tensor4('embds')              # : (nblanks, minibatch_size, 2, embedding_dim)
        blankidxs = T.imatrix('indexes')    # : (nblanks, minibatch_size)
        masks = T.matrix('masks')           # : (maxnblanks, minibatch_size)

        x_embds = embeddings[x.flatten()].reshape((x.shape[0], x.shape[1], embedding_dim))

        self.in2lstm = HiddenLayer(embedding_dim, lstm_dim, 'tanh')
        self.lstm = LSTMLayer(lstm_dim)
        self.lstm2gate = HiddenLayer(lstm_dim, embedding_dim, None)
        self.e2gate = HiddenLayer(embedding_dim, embedding_dim, None)

        prelstm = self.in2lstm.apply(x_embds)
        hs = self.lstm.apply(prelstm)           # : (seq_len, minibatch_size, lstm_dim)

        def _step(ith_blank):
            return hs[ith_blank, T.arange(hs.shape[1])]

        indexed_hs, _ = theano.scan(            # : (nblanks, minibatch_size, lstm_dim)
                    _step, blankidxs, [],
                    profile = _doProfile)

        gated_es = e

        if use_gate:
            act = {'sigmoid': T.nnet.sigmoid, 'relu': lambda x : T.maximum(0, x), None: lambda x : x}[gate_activation]
            gated_es = act(self.lstm2gate.apply(indexed_hs).dimshuffle(0, 1, 'x', 2) + self.e2gate.apply(e))
            gated_es = gated_es * e

        self.W = theano.shared(numpy.random.uniform(-.1, .1, (embedding_dim, lstm_dim)).astype(config.floatX), 'W')
        self.b = theano.shared(numpy.float32(0.), 'b')

        eC = T.dot(gated_es, self.W)
        crs = T.sum(eC * indexed_hs.dimshuffle(0, 1, 'x', 2), axis = 3)
        pred = T.nnet.sigmoid(crs.dimshuffle(0, 1, 2, 'x') + self.b).flatten(3)

        error = T.argmax(pred * masks.dimshuffle(0, 1, 'x'), axis = 2).sum()
        cost = T.mean(T.sum(T.nnet.binary_crossentropy(pred, y) * masks.dimshuffle(0, 1, 'x'), axis = [0, 2]) / masks.sum(axis = 0))

        params = self.in2lstm.params + self.lstm.params + [embeddings, self.W, self.b]

        if use_gate:
            params += self.lstm2gate.params + self.e2gate.params

        self.params = params
        grads = T.grad(cost, params)
        updates = optimization_method(params, grads)

        self.learn = theano.function([x, y, e, blankidxs, masks], [cost, error], updates = updates, profile = _doProfile)
        self.test = theano.function([x, e, blankidxs, masks], [error])

    def save_params(self, path):
        cPickle.dump([i.get_value() for i in self.params], open(path, 'w'), -1)

    def load_params(self, path):
        values = cPickle.load(open(path, 'r'))
        for v, p in zip(values, self.params):
            p.set_value(v)
