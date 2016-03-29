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
import uuid
import os
from model import *

config = theano.config
random.seed(1234)

def load_embeddings(path, size, dim, random):
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

def get_x_batch(docs, batch, batch_size):
    batch_xs = docs[batch * batch_size : batch * batch_size + batch_size]
    return batch_xs, len(batch_xs)

def get_e_batch(lexs, batch, batch_size):
    batch_es = lexs[batch * batch_size : batch * batch_size + batch_size]
    nblanks = [len(i) for i in batch_es]
    return batch_es, nblanks, max(nblanks)

def prep_x_batch(max_len, max_nblanks, actual_size, batch_xs):
    x = numpy.zeros((max_len, actual_size), dtype = 'int32')
    bid = numpy.zeros((max_nblanks, actual_size), dtype = 'int32')

    for j, xi in enumerate(batch_xs):
        x[:xi.shape[0], j] = xi
        nonzeros = numpy.nonzero(xi == 60001)[0]
        bid[:nonzeros.shape[0], j] = nonzeros

    return x, bid

def prep_y_batch(e):
    y = numpy.zeros(e.shape[:3]).astype(config.floatX)
    y[:, :, 0] = 1
    return y

def prep_e_batch(max_nblanks, actual_size, batch_es):
    e = numpy.zeros((max_nblanks, actual_size, 2, 300), dtype = 'float32')
    masks = numpy.zeros((max_nblanks, actual_size), dtype = 'float32')

    for j, ei in enumerate(batch_es):
        e[:len(ei), j] = ei
        masks[:len(ei), j] = 1

    return e, masks

def launch_exp(settings):
    exp_dir = settings['datapath'] + 'result/{0}/'.format(uuid.uuid1())
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    json.dump(settings, open(exp_dir + 'settings.json', 'w'), indent = 4)
    print 'Settings:', settings, '\n'

    float32 = lambda x : numpy.float32(x)

    train = cPickle.load(open(settings['datapath'] + 'train/{0}/train.pkl'.format(settings['lex_version']), 'r'))
    train_docs = train['x']
    train_lexs = train['e']
    num_train = len(train_docs)
    valid = cPickle.load(open(settings['datapath'] + 'valid/{0}/valid.pkl'.format(settings['lex_version']), 'r'))
    valid_docs = valid['x']
    valid_lexs = valid['e']
    num_valid = len(valid_docs)

    num_epochs = settings['num_epochs']
    batch_size = settings['batch_size']
    lr = float32(settings['lr_rate'])

    embeddings = load_embeddings(settings['datapath'], settings['vocab_size'], settings['embedding_dim'], settings['random_init'])
    if settings['optimization_method'] == 'sgd':
        optimization_method = sgd(lr)
    elif settings['optimization_method'] == 'adam':
        optimization_method = Adam(lr,
                                   settings['adam_beta1'],
                                   settings['adam_beta2'],
                                   settings['adam_epsilon'])
    else:
        raise ValueError

    model = Predictor(embeddings,
                      embedding_dim = settings['embedding_dim'], lstm_dim = settings['lstm_dim'],
                      use_gate = settings['use_gate'], gate_activation = settings['gate_activation'],
                      optimization_method = optimization_method)

    exp_results = {'train_results': {'costs': [], 'errors': []}, 'valid_results': []}
    best_valid_err = 1.

    print 'Start training ......'
    for epoch in range(0, num_epochs):
        train_docs, train_lexs = shuffle_data(train_docs, train_lexs)
        epoch_cost, epoch_error, epoch_nblanks = 0, 0, 0
        epoch_t0 = time.time()

        for batch in range(0, num_train / batch_size + 1):
            # the Xs #
            batch_xs, actual_size = get_x_batch(train_docs, batch, batch_size)
            if actual_size == 0:
                continue

            # the blank embeddings and masks #
            batch_es, nblanks, max_nblanks = get_e_batch(train_lexs, batch, batch_size)
            e, masks = prep_e_batch(max_nblanks, actual_size, batch_es)

            # the Xs into an array, and blank indexes #
            max_len = max(len(i) for i in batch_xs)
            x, bid = prep_x_batch(max_len, max_nblanks, actual_size, batch_xs)

            # the Ys #
            y = prep_y_batch(e)

            # train on mini-batch #
            cost, err = model.learn(x, y, e, bid, masks)

            # cost, error #
            epoch_cost += cost * actual_size
            epoch_error += err
            epoch_nblanks += sum(nblanks)

        epoch_elapsed = time.time() - epoch_t0
        print '# epochs: {0}'.format(epoch + 1)
        print '\t cost: {0}; error: {1}; time elapsed: {2} s\n'.format(epoch_cost / num_train, epoch_error * 1. / epoch_nblanks, numpy.round(epoch_elapsed, 2))

        exp_results['train_results']['costs'].append(epoch_cost / num_train)
        exp_results['train_results']['errors'].append(epoch_error * 1. / epoch_nblanks)

        if (epoch + 1) % settings['valid_freq'] == 0:
            valid_docs, valid_lexs = shuffle_data(valid_docs, valid_lexs)
            valid_error, valid_nblanks = 0, 0
            valid_t0 = time.time()

            print '\t Testing on validation set ......'
            for batch in range(0, num_valid / batch_size + 1):
                # the Xs #
                batch_xs, actual_size = get_x_batch(valid_docs, batch, batch_size)
                if actual_size == 0:
                    continue

                # the blank embeddings and masks #
                batch_es, nblanks, max_nblanks = get_e_batch(valid_lexs, batch, batch_size)
                e, masks = prep_e_batch(max_nblanks, actual_size, batch_es)

                # the Xs into an array, and blank indexes #
                max_len = max(len(i) for i in batch_xs)
                x, bid = prep_x_batch(max_len, max_nblanks, actual_size, batch_xs)

                # test on mini-batch #
                err = model.test(x, e, bid, masks)[0]

                # error on validation set #
                valid_error += err
                valid_nblanks += sum(nblanks)

            valid_elapsed = time.time() - valid_t0
            print '\t\t error: {0}; time elapsed: {1} s\n'.format(valid_error * 1. / valid_nblanks, numpy.round(valid_elapsed, 2))

            exp_results['valid_results'].append((epoch + 1, valid_error * 1. / valid_nblanks))

            model.save_params(exp_dir + 'curr_train_model.pkl')
            if (valid_error * 1. / valid_nblanks) < best_valid_err:
                best_valid_err = valid_error * 1. / valid_nblanks
                model.save_params(exp_dir + 'best_valid_model.pkl')
                print '\t\t [NEW best validation error: {0}, save current model]\n'.format(best_valid_err)

    json.dump(exp_results, open(exp_dir + 'exp_results.json', 'w'), indent = 4)
    print 'Training complete'
    print '\t - training cost: {0}; training error: {1}'.format(exp_results['train_results']['costs'][-1], exp_results['train_results']['errors'][-1])
    print '\t - best validation error: {0}'.format(best_valid_err)

if __name__ == '__main__':
    settings = {
        'datapath': '/scratch/data/freelink/',      # path to dataset
        'lex_version': 'name',                      # lexical embedding version
        'valid_freq': 5,                            # frequency to test on validation set
        'vocab_size': 60000 + 2,                    # vocabulary size
        'random_init': False,                       # random initialization of word embeddings
        'num_epochs': 100,                          # number of training epochs
        'batch_size': 32,                           # size of mini-batch
        'lr_rate': 0.005,                           # learning rate
        'embedding_dim': 300,                       # word embedding dimension
        'lstm_dim': 128,                            # lstm layer dimension
        'use_gate': False,                          # use filter gate
        'gate_activation': 'sigmoid',               # gate activation function
        'optimization_method': 'sgd',               # either adam or sgd
        'adam_beta1':0.9,                           # 1st adam hyperparameter
        'adam_beta2':0.999,                         # 2nd adam hyperparameter
        'adam_epsilon':1e-4                         # 3rd adam hyperparameter
    }

    #####################
    # launch experiment #
    #####################
    launch_exp(settings)
