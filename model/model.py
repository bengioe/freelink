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

config = theano.config
config.compute_test_value = 'off'
_doProfile = False

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
    def __init__(self, embeddings, embedding_dim = 300, lstm_dim = 128, use_gate = False, gate_activation='sigmoid'):
        lr = T.scalar('lr')                 # : scalar
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

        self.lstm2pred = HiddenLayer(lstm_dim, 1, None)
        self.e2pred = HiddenLayer(embedding_dim, 1, None)

        prelstm = self.in2lstm.apply(x_embds)
        hs = self.lstm.apply(prelstm)           # : (seq_len, minibatch_size, lstm_dim)

        def _step(ith_blank):
            return hs[ith_blank, T.arange(hs.shape[1])]

        indexed_hs, _ = theano.scan(            # : (nblanks, minibatch_size, lstm_dim)
                    _step, blankidxs, [],
                    profile = _doProfile)

        self.crsW = theano.shared(numpy.random.uniform(-.1, .1, (embedding_dim, lstm_dim)).astype(config.floatX), 'crsW')
        eC = T.dot(e, self.crsW)
        crs = T.sum(eC * indexed_hs.dimshuffle(0, 1, 'x', 2), axis = 3)

        hpred = self.lstm2pred.apply(indexed_hs)        # : (nblanks, minibatch_size, 1)
        if use_gate:
            act = {'sigmoid': T.nnet.sigmoid, 'relu': lambda x : T.maximum(0, x), None: lambda x : x}[gate_activation]
            classifier_e_input = act(self.lstm2gate.apply(indexed_hs).dimshuffle(0, 1, 'x', 2) + self.e2gate.apply(e))
            classifier_e_input = classifier_e_input * e
        else:
            classifier_e_input = e
        epred = self.e2pred.apply(classifier_e_input)   # : (nblanks, minibatch_size, 2, 1)

        pred = None                                     # : (nblanks, minibatch_size, 2)
        if use_gate:
            pred = T.nnet.sigmoid(hpred.dimshuffle(0, 1, 'x', 2) + epred).flatten(3)
        else:
            pred = T.nnet.sigmoid(hpred.dimshuffle(0, 1, 'x', 2) + epred + crs.dimshuffle(0, 1, 2, 'x')).flatten(3)

        error = T.argmax(pred * masks.dimshuffle(0, 1, 'x'), axis = 2).sum()
        cost = T.mean(T.sum(T.nnet.binary_crossentropy(pred, y) * masks.dimshuffle(0, 1, 'x'), axis = [0, 2]) / masks.sum(axis = 0))

        params = self.in2lstm.params + self.lstm.params + self.lstm2pred.params + self.e2pred.params + [embeddings]
        if use_gate:
            params += self.lstm2gate.params + self.e2gate.params
        else:
            params += [self.crsW]

        self.params = params
        grads = T.grad(cost, params)
        updates = [(i, i - lr * gi) for i, gi in zip(params, grads)]

        self.learn = theano.function([x, y, e, lr, blankidxs, masks], [cost, error], updates = updates, profile = _doProfile)
        self.test = theano.function([x, e, blankidxs, masks], [error])

    def save_params(self, path):
        cPickle.dump([i.get_value() for i in self.params], open(path, 'w'), -1)

    def load_params(self, path):
        values = cPickle.load(open(path, 'r'))
        for v, p in zip(values, self.params):
            p.set_value(v)
