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
    def __init__(self, embeddings, embedding_dim=300, lstm_dim=64):

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

        x_embds = embeddings[x.flatten()].reshape((x.shape[0], x.shape[1], embedding_dim))

        self.in2lstm = HiddenLayer(embedding_dim, lstm_dim, 'tanh')
        self.lstm = LSTMLayer(lstm_dim)
        self.lstm2pred = HiddenLayer(lstm_dim, 1, None)
        self.e2pred = HiddenLayer(embedding_dim, 1 , None)

        prelstm = self.in2lstm.apply(x_embds)
        hs = self.lstm.apply(prelstm) # (seq len, minibatch_size, lstm_dim)

        def _step(ith_blank):
            return hs[ith_blank, T.arange(hs.shape[1])]

        # (nblanks, minibatch_size, lstm_dim)
        indexed_hs, _ = theano.scan(_step, blankidxs,[], profile=_doProfile)

        self.crsW = theano.shared(numpy.random.uniform(-.1,.1, (embedding_dim, lstm_dim)), 'crsW')
        eC = T.dot(e, self.crsW)

        crs = T.sum(eC * indexed_hs.dimshuffle(0,1,'x',2), axis=3)

        hpred = self.lstm2pred.apply(indexed_hs)
        epred = self.e2pred.apply(e)

        pred = T.nnet.sigmoid(hpred.dimshuffle(0,1,'x',2) + epred +
                        crs.dimshuffle(0,1,2,'x')).flatten(3)

        cost = T.mean(T.nnet.binary_crossentropy(pred, y))

        params = [self.crsW] + self.in2lstm.params + self.lstm.params + \
            self.lstm2pred.params + self.e2pred.params + [embeddings]

        grads = T.grad(cost, params)

        updates = [(i, i - lr * gi) for i,gi in zip(params, grads)]
        self.learn = theano.function([x, y, e ,lr, blankidxs], [cost], updates=updates, profile=_doProfile)

def shuffle_data(x, x_p):
    index = range(0, len(x))
    random.shuffle(index)
    return [x[i] for i in index], [x_p[j] for j in index]

def main():
    import time
    print "Loading data"
    random.seed(1234)
    t0 = time.time()
    path = '/scratch/data/freelink/valid/{0}/'.format(sys.argv[1])
    train = cPickle.load(open(path + 'valid.pkl', 'r'))
    documents = train['x'][:100]
    blanks = train['e'][:100]
    print "took",int(time.time()-t0), "seconds"

    nexamples = len(documents)

    embeddings = theano.shared(numpy.random.random((60002, 300)).astype(config.floatX))
    print "Making model"
    theano.config.compute_test_value = 'raise'
    model = Model(embeddings)
    f = lambda x : numpy.float32(x)
    x = numpy.int32(numpy.random.random((100, 7)))
    y = f(numpy.random.random((4, 7, 2)))
    e = f(numpy.random.random((4, 7, 2, 300)))
    bid = numpy.int32(numpy.random.random((4, 7)))
    lr = f(0.1)

    print "Training"
    for epoch in range(0, 100):
        documents, blanks = shuffle_data(documents, blanks)
        epoch_cost = 0
        t0 = time.time()
        print '# examples:', nexamples

        for i in range(0, nexamples):
            x = documents[i][:,None]
            '''
            print x.shape[0]
            if x.shape[0] > 2000:
                print "skip"
                continue
            '''
            e = numpy.float32(blanks[i])[:,None,:,:]
            y = numpy.zeros(e.shape[:3]).astype(config.floatX)
            y[:,:,0] = 1
            bid = numpy.int32(numpy.nonzero(numpy.int32(documents[i]) == 60001)[0])[:, None]

            cost, = model.learn(x, y, e, lr, bid)
            '''
            print i, cost
            '''
            epoch_cost += cost
        print '# epochs: {0}; cost: {1}'.format(i + 1, epoch_cost / nexamples)


if __name__ == "__main__":
    main()
