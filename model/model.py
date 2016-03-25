'''
created on Mar 23, 2016

@author: leolong
'''

import theano.tensor as T
import theano
import numpy
import numpy.linalg
import cPickle
import sys
import random

config = theano.config
_doProfile = False
theano.config.compute_test_value = 'off'

def ortho_matrix(ndim):
    W = numpy.random.randn(ndim, ndim)
    u,s,v = numpy.linalg.svd(W)
    return u.astype(config.floatX)

class LSTMLayer:
    def __init__(self, ndim):
        W = numpy.concatenate([ortho_matrix(ndim) for i in range(4)], axis=1)
        U = numpy.concatenate([ortho_matrix(ndim) for i in range(4)], axis=1)

        b = numpy.zeros(4*ndim).astype(config.floatX)

        self.params = [theano.shared(i) for i in [W,U,b]]
        self.ndim = ndim

    def apply(self, x):
        W, U, b = self.params
        ndim = self.ndim

        def _slice(x,n,dim):
            return x[:,n*dim:(n+1)*dim]

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
                outputs_info=[T.alloc(numpy.float32(0.), x.shape[1], ndim),
                              T.alloc(numpy.float32(0.), x.shape[1], ndim)]
                              , profile=_doProfile)
        return rval[0]


class HiddenLayer:
    def __init__(self, nin, nout, activation):
        if activation in ['tanh','sigmoid', None]:
            k = numpy.sqrt(6./(nin+nout))
        elif activation == 'relu':
            k = numpy.sqrt(6./nin)
        self.W = theano.shared(numpy.random.uniform(-k,k, (nin,nout)).astype(config.floatX))
        self.b = theano.shared(numpy.zeros(nout).astype(config.floatX))
        self.activation = {'relu':lambda x:T.maximum(0,x), "tanh":T.tanh, 'sigmoid':T.nnet.sigmoid, None:lambda x:x}[activation]
        self.params = [self.W, self.b]
    def apply(self, x):

        return self.activation(T.dot(x,self.W) + self.b)


class Model:
    def __init__(self, embeddings, embedding_dim=300, lstm_dim=128, useGate=False, gate_activation='sigmoid'):
        f = lambda x: numpy.float32(x)
        _x = numpy.int32(numpy.random.random((100, 7)))
        _y = f(numpy.random.random((4, 7, 2)))
        _e = f(numpy.random.random((4, 7, 2, 300)))
        _bid = numpy.int32(numpy.random.random((4, 7)))
        lr = T.scalar('lr')
        lr.tag.test_value = f(0.1)
        x = T.imatrix('input') # : (seq len, minibatch_size)
        x.tag.test_value = _x
        y = T.tensor3('targets') # : (nblanks, minibatch_size, 2)
        y.tag.test_value = _y
        e = T.tensor4('embds') # : (nblanks, minibatch_size, 2, embedding_dim)
        e.tag.test_value = _e
        blankidxs = T.imatrix('indexes') # (nblanks, minibatch_size)
        blankidxs.tag.test_value = _bid
        masks = T.matrix('masks') # (maxnblanks, minibatch_size)
        masks.tag.test_value = numpy.zeros((4, 7), dtype='float32')

        x_embds = embeddings[x.flatten()].reshape((x.shape[0], x.shape[1], embedding_dim))

        self.in2lstm = HiddenLayer(embedding_dim, lstm_dim, 'tanh')
        self.lstm = LSTMLayer(lstm_dim)
        self.lstm2pred = HiddenLayer(lstm_dim, 1, None)
        self.e2pred = HiddenLayer(embedding_dim, 1 , None)
        self.lstm2gate = HiddenLayer(lstm_dim, embedding_dim, None)
        self.e2gate = HiddenLayer(embedding_dim, embedding_dim, None)

        prelstm = self.in2lstm.apply(x_embds)
        hs = self.lstm.apply(prelstm) # (seq len, minibatch_size, lstm_dim)

        def _step(ith_blank):
            return hs[ith_blank, T.arange(hs.shape[1])]

        # (nblanks, minibatch_size, lstm_dim)
        indexed_hs, _ = theano.scan(_step, blankidxs,[], profile=_doProfile)

        self.crsW = theano.shared(numpy.random.uniform(-.1,.1, (embedding_dim, lstm_dim)), 'crsW')
        eC = T.dot(e, self.crsW)

        crs = T.sum(eC * indexed_hs.dimshuffle(0,1,'x',2), axis=3)

        hpred = self.lstm2pred.apply(indexed_hs) #(nblanks, mbs, 1)
        if useGate:
            act = {'relu':lambda x:T.maximum(0,x), 'sigmoid':T.nnet.sigmoid, None:lambda x:x}[gate_activation]
            classifier_e_input = act(self.lstm2gate.apply(indexed_hs).dimshuffle(0,1,'x',2) + self.e2gate.apply(e))
            classifier_e_input = classifier_e_input * e
        else:
            classifier_e_input = e
        epred = self.e2pred.apply(classifier_e_input) # (nblanks, mbs, 2, 1)

        # pred: (nblanks, mbs, 2)
        if useGate:
            pred = T.nnet.sigmoid(hpred.dimshuffle(0,1,'x',2) + epred).flatten(3)
        else:
            pred = T.nnet.sigmoid(hpred.dimshuffle(0,1,'x',2) + epred +
                            crs.dimshuffle(0,1,2,'x')).flatten(3)
        error = T.argmax(pred * masks.dimshuffle(0,1,'x'), axis=2).sum()
        cost = T.mean(T.sum(T.nnet.binary_crossentropy(pred, y) * masks.dimshuffle(0,1,'x'), axis=[0,2]) / masks.sum(axis=0))

        params = self.in2lstm.params + self.lstm.params + \
            self.lstm2pred.params + self.e2pred.params + [embeddings]
        if useGate:
            params += self.lstm2gate.params + self.e2gate.params
        else:
            params += [self.crsW]
        self.params = params

        grads = T.grad(cost, params)

        updates = [(i, i - lr * gi) for i,gi in zip(params, grads)]
        self.learn = theano.function([x, y, e ,lr, blankidxs, masks], [cost, error], updates=updates, profile=_doProfile)
        self.test = theano.function([x, y, e, blankidxs, masks], [error])

    def save_params(self, path):
        cPickle.dump([i.get_value() for i in self.params], open(path,'w'), -1)
    def load_params(self, path):
        values = cPickle.load(open(path,'r'))
        for v,p in zip(values, self.params):
            p.set_value(v)

def shuffle_data(x, x_p):
    index = range(0, len(x))
    random.shuffle(index)
    return [x[i] for i in index], [x_p[j] for j in index]

def main():
    import time
    print "Loading data"
    random.seed(1234)
    t0 = time.time()
    results_path = 'results.pkl'
    params_path = 'params.pkl'
    path = '/scratch/data/freelink/valid/{0}/'.format(sys.argv[1])
    train = cPickle.load(open(path + 'valid.pkl', 'r'))
    documents = train['x'][:1000]
    blanks = train['e'][:1000]
    print "took",int(time.time()-t0), "seconds"

    exp_results = {"train_results":{"errors":[], "costs":[]}, "valid_results":{}}

    nexamples = len(documents)

    embeddings = theano.shared(numpy.random.random((60002, 300)).astype(config.floatX))
    print "Making model"
    model = Model(embeddings)
    f = lambda x : numpy.float32(x)
    x = numpy.int32(numpy.random.random((100, 7)))
    y = f(numpy.random.random((4, 7, 2)))
    e = f(numpy.random.random((4, 7, 2, 300)))
    bid = numpy.int32(numpy.random.random((4, 7)))
    lr = f(0.05)
    minibatch_size = 16

    print "Training"
    for epoch in range(0, 100):
        documents, blanks = shuffle_data(documents, blanks)
        epoch_cost = 0
        epoch_error = 0
        epoch_nblanks = 0
        t0 = time.time()
        print '# examples:', nexamples

        for mb in range(0, nexamples / minibatch_size + 1):
            # the Xs
            minibatch_xs = documents[mb*minibatch_size:mb*minibatch_size+minibatch_size]
            actual_minibatch_size = mbs = len(minibatch_xs)
            if mbs == 0: continue

            # the blank embeddings and masks
            mb_es = blanks[mb*minibatch_size:mb*minibatch_size+minibatch_size]
            nblanks = [len(i) for i in mb_es]
            maxnblanks = max(nblanks)
            e = numpy.zeros((maxnblanks, mbs, 2, 300), dtype='float32')
            masks = numpy.zeros((maxnblanks, mbs), dtype='float32')
            for j,ei in enumerate(mb_es):
                e[:len(ei),j] = ei
                masks[:len(ei),j] = 1

            # the Xs into an array, and blank indexes
            maxlen = max(len(i) for i in minibatch_xs)
            x = numpy.zeros((maxlen, mbs),dtype='int32')
            bid = numpy.zeros((maxnblanks, mbs), dtype='int32')
            for j,xi in enumerate(minibatch_xs):
                x[:xi.shape[0],j] = xi
                nonzeros = numpy.nonzero(xi == 60001)[0]
                bid[:nonzeros.shape[0],j] = nonzeros

            y = numpy.zeros(e.shape[:3]).astype(config.floatX)
            y[:,:,0] = 1

            cost,err = model.learn(x, y, e, lr, bid, masks)
            print mb, nexamples / minibatch_size, cost, err
            epoch_error += err
            epoch_nblanks += sum(nblanks)
            epoch_cost += cost * mbs

        epochtime = time.time() - t0
        print '# epochs: {0}; cost: {1}; error: {3} took: {2}s'.format(epoch + 1, epoch_cost / nexamples, numpy.round(epochtime,2),
            epoch_error * 1. / epoch_nblanks)

        exp_results['train_results']['errors'].append(epoch_error * 1. / epoch_nblanks)
        exp_results['train_results']['costs'].append(epoch_cost / nexamples)

        cPickle.dump(exp_results, open(results_path,'w'), -1)
        model.save_params(params_path)

if __name__ == "__main__":
    main()
