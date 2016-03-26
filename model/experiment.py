'''
created on Mar 25, 2016

@author: leolong
'''

import theano
import theano.tensor as T
import numpy
import cPickle
import random
import sys
import json
from model import *

config = theano.config
random.seed(1234)

def load_embeddings(path, size, random):
    if random:
        return theano.shared(numpy.random.random((size, 300)).astype(config.floatX))

    glove = cPickle.load(open('/scratch/data/embedding/' + 'glove.840B.300d.pkl', 'r'))
    vocabulary = json.load(open(path + 'vocabulary.json', 'r'))

    v_sorted = sorted(vocabulary.items(), key = lambda item : item[1])
    word_vectors = []

    for item in v_sorted:
        if item[0] in glove['glove']:
            word_vectors += [glove['glove'][item[0]]]
        else:
            word_vectors += [glove['mean']]

    unk, mark = glove['mean'], numpy.zeros(300)
    word_vectors += [unk, mark]
    return theano.shared(numpy.vstack(word_vectors).astype(config.floatX))

def shuffle_data(docs, lexs):
    indexes = range(0, len(docs))
    random.shuffle(indexes)
    return [docs[i] for i in indexes], [lexs[j] for j in indexes]

def launch(datapath = '/scratch/data/freelink/', lex_version = 'name', vocab_size = 60002, random_emb = True,
            batch_size = 16, l_rate = 0.05, num_epochs = 100):
    float32 = lambda x : numpy.float32(x)

    train = cPickle.load(open(datapath + 'train/{0}/train.pkl'.format(lex_version), 'r'))
    train_docs = train['x']
    train_lexs = train['e']
    ndocs = len(train_docs)
    lr = float32(l_rate)

    valid = cPickle.load(open(datapath + 'valid/{0}/valid.pkl'.format(lex_version), 'r'))
    valid_docs = valid['x']
    valid_lexs = valid['e']

    model = Predictor(load_embeddings(datapath, vocab_size, random_emb))
    exp_results = {'train_results': {'costs': [], 'errors': []}, 'valid_results': {}}

    print 'Start training ......'
    for epoch in range(0, num_epochs):
        train_docs, train_lexs = shuffle_data(train_docs, train_lexs)
        epoch_cost, epoch_error, epoch_nblanks = 0, 0, 0
        t_0 = time.time()

        for batch in range(0, ndocs / batch_size + 1):
            # the Xs #
            batch_xs = train_docs[batch * batch_size : batch * batch_size + batch_size]
            actual_size = len(batch_xs)
            if actual_size == 0:
                continue

            # the blank embeddings and masks #
            batch_es = train_lexs[batch * batch_size : batch * batch_size + batch_size]
            nblanks = [len(i) for i in batch_es]
            max_nblanks = max(nblanks)

            e = numpy.zeros((max_nblanks, actual_size, 2, 300), dtype = 'float32')
            masks = numpy.zeros((max_nblanks, actual_size), dtype = 'float32')
            for j, ei in enumerate(batch_es):
                e[:len(ei), j] = ei
                masks[:len(ei), j] = 1

            # the Xs into an array, and blank indexes #
            max_len = max(len(i) for i in batch_xs)

            x = numpy.zeros((max_len, actual_size), dtype = 'int32')
            bid = numpy.zeros((max_nblanks, actual_size), dtype = 'int32')
            for j, xi in enumerate(batch_xs):
                x[:xi.shape[0], j] = xi
                nonzeros = numpy.nonzero(xi == 60001)[0]
                bid[:nonzeros.shape[0], j] = nonzeros

            # the Ys #
            y = numpy.zeros(e.shape[:3]).astype(config.floatX)
            y[:, :, 0] = 1

            # train on mini-batch #
            cost, err = model.learn(x, y, e, lr, bid, masks)
            # cost, error #
            epoch_cost += cost * actual_size
            epoch_error += err
            epoch_nblanks += sum(nblanks)

        '''

        Here: Compute validation set results

        '''

        time_elapsed = time.time() - t_0
        print '# epochs: {0}'.format(epoch + 1)
        print '\t cost: {0}; error: {1}; time elapsed: {2} s'.format(epoch_cost / ndocs, epoch_error / 1. * epoch_nblanks, numpy.round(time_elapsed, 2))

        exp_results['train_results']['costs'].append(epoch_cost / ndocs)
        exp_results['train_results']['errors'].append(epoch_error * 1. / epoch_nblanks)

if __name__ == '__main__':
    launch(lex_version = 'name', random_emb = False)
