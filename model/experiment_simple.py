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
from model_simple import *

config = theano.config
random.seed(1234)

def load_embeddings(random_init, voc_size, path):
    if random_init:
        return theano.shared(numpy.random.uniform(-.1, .1, (voc_size + 2, 300)).astype(config.floatX))

    glove_embeddings = cPickle.load(open(path + 'glove.840B.300d.pkl', 'r'))
    word2idx = json.load(open(path + 'word2idx.json', 'r'))
    w_sorted = sorted(word2idx.items(), key = lambda item : item[1])

    word_vectors = []
    for item in w_sorted:
        if item[0] in glove_embeddings['glove']:
            word_vectors.append(glove_embeddings['glove'][item[0]])
        else:
            word_vectors.append(glove_embeddings['mean'])
    word_vectors.append(glove_embeddings['mean'])
    word_vectors.append(numpy.random.uniform(-.1, .1, 300).astype(config.floatX))

    return theano.shared(numpy.vstack(word_vectors).astype(config.floatX))

def shuffle_data(docs, lexs):
    indexes = range(0, len(docs))
    random.shuffle(indexes)

    return [docs[i] for i in indexes], [lexs[j] for j in indexes]

def launch_exp(settings):
    exp_dir = settings['datapath'] + 'result/{0}/'.format(uuid.uuid1())
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    json.dump(settings, open(exp_dir + 'settings.json', 'w'), indent = 4)
    print 'Settings:', settings, '\n'

    if 1: # create random data for testing purposes
        train_docs = [[numpy.random.randint(0,1000) for i in range(numpy.random.randint(0,600))] for _ in range(1000)]
        if settings['use_definitions']:
            train_defs = [[numpy.random.randint(0,1000) for i in range(numpy.random.randint(0,60))] for _ in range(1000)]
            valid_defs = train_defs
        else:
            train_es = numpy.float32([numpy.random.random((300,)) for _ in range(1000)])
            valid_es = train_es
        train_ys = [numpy.random.randint(0,2) for _ in range(1000)]
        num_train = 1000
        
        valid_docs = train_docs
        valid_ys = train_ys
        num_valid = 1000
    else:
        # this needs to be fixed for new data
        train = cPickle.load(open(settings['datapath'] + 'train/train_{0}.pkl'.format(settings['lex_version']), 'r'))
        train_docs = train['x']
        train_lexs = train['e']
        num_train = len(train_docs)

        valid = cPickle.load(open(settings['datapath'] + 'valid/valid_{0}.pkl'.format(settings['lex_version']), 'r'))
        valid_docs = valid['x']
        valid_lexs = valid['e']
        num_valid = len(valid_docs)

    num_epochs = settings['num_epochs']
    batch_size = settings['batch_size']
    lr = numpy.float32(settings['lr_rate'])

    try:
        voc_size = len(json.load(open(settings['datapath'] + 'word2idx.json', 'r')))
    except Exception as e:
        print("Couldn't read word2idx file", e)
        voc_size = 30000
    embeddings = load_embeddings(settings['random_init'], voc_size, settings['datapath'])

    if settings['optimization_method'] == 'sgd':
        optimization_method = sgd(lr)
    elif settings['optimization_method'] == 'adam':
        optimization_method = Adam(lr,
                                   settings['adam_beta1'],
                                   settings['adam_beta2'],
                                   settings['adam_epsilon'])
    else:
        raise ValueError

    model = Predictor(embeddings, lstm_dim = settings['lstm_dim'],
                      use_gate = settings['use_gate'], gate_activation = settings['gate_activation'],
                      crs_term = settings['crs_term'], optimization_method = optimization_method,
                      use_definitions = settings['use_definitions'])

    exp_results = {'train_results': {'costs': [], 'errors': []}, 'valid_results': []}
    best_valid_err = 1.

    print 'Start training ......'
    for epoch in range(0, num_epochs):
        if settings['use_definitions']:
            train_docs, train_defs = shuffle_data(train_docs, train_defs)
        else:
            train_docs, train_es = shuffle_data(train_docs, train_es)
        epoch_cost, epoch_error = 0, 0
        epoch_t0 = time.time()

        def zero_pad(l):
            max_len = max(len(i) for i in l)
            z = numpy.zeros((len(l), max_len),dtype='int32')
            masks = numpy.zeros((len(l), max_len),dtype='int32')
            for i,s in enumerate(l):
                z[i,:len(s)] = s
                masks[i,:len(s)] = 1
            return z.T, masks.T
        
        for batch in range(0, num_train / batch_size + 1):
            docs, docs_masks = zero_pad(train_docs[batch*batch_size: (batch+1)*batch_size])
            actual_size = docs.shape[0]
            if actual_size == 0:
                continue

            
            if settings['use_definitions']:
                defs, defs_masks = zero_pad(train_defs[batch*batch_size: (batch+1)*batch_size])
            else:
                es = train_es[batch*batch_size: (batch+1)*batch_size]
            ys = train_ys[batch*batch_size: (batch+1)*batch_size]

            if settings['use_definitions']:
                cost, err = model.learn(docs, ys, defs, docs_masks, defs_masks)
            else:
                cost, err = model.learn(docs, ys, es, docs_masks)
            
            epoch_cost += cost * actual_size
            epoch_error += err

        epoch_elapsed = time.time() - epoch_t0
        print '# epochs: {0}'.format(epoch + 1)
        print '\t cost: {0}; error: {1}; time elapsed: {2} s\n'.format(epoch_cost / num_train, epoch_error * 1. / num_train, numpy.round(epoch_elapsed, 2))

        exp_results['train_results']['costs'].append(epoch_cost / num_train)
        exp_results['train_results']['errors'].append(epoch_error * 1. / num_train)
        model.save_params(exp_dir + 'curr_train_model.pkl')

        if (epoch + 1) % settings['valid_freq'] == 0:
            valid_error = 0
            valid_t0 = time.time()

            print '\t Testing on validation set ......'
            for batch in range(0, num_train / batch_size + 1):
                docs, docs_masks = zero_pad(valid_docs[batch*batch_size: (batch+1)*batch_size])
                actual_size = docs.shape[0]
                if actual_size == 0:
                    continue


                if settings['use_definitions']:
                    defs, defs_masks = zero_pad(valid_defs[batch*batch_size: (batch+1)*batch_size])
                else:
                    es = valid_es[batch*batch_size: (batch+1)*batch_size]
                ys = valid_ys[batch*batch_size: (batch+1)*batch_size]

                if settings['use_definitions']:
                    err, = model.test(docs, ys, defs, docs_masks, defs_masks)
                else:
                    err, = model.test(docs, ys, es, docs_masks)

                valid_error += err


            valid_elapsed = time.time() - valid_t0
            print '\t\t error: {0}; time elapsed: {1} s\n'.format(valid_error * 1. / num_valid, numpy.round(valid_elapsed, 2))

            exp_results['valid_results'].append((epoch + 1, valid_error * 1. / num_valid))
            if (valid_error * 1. / num_valid) < best_valid_err:
                best_valid_err = valid_error * 1. / num_valid
                model.save_params(exp_dir + 'best_valid_model.pkl')
                print '\t\t [NEW best validation error: {0}, save current model]\n'.format(best_valid_err)

        # save current experiment results #
        json.dump(exp_results, open(exp_dir + 'exp_results.json', 'w'), indent = 4)

    print 'Training complete'
    print '\t - training cost: {0}; training error: {1}'.format(exp_results['train_results']['costs'][-1], exp_results['train_results']['errors'][-1])
    print '\t - best validation error: {0}'.format(best_valid_err)

if __name__ == '__main__':
    settings = {
        'datapath': sys.argv[1],                    # path to dataset
        'lex_version': 'flex',                      # lexical embedding version
        'valid_freq': 1,                            # frequency to test on validation set
        'random_init': True,                       # random initialization of word embeddings
        'num_epochs': 20,                           # number of training epochs
        'batch_size': 128,                          # size of mini-batch
        'lr_rate': 0.0005,                          # learning rate
        'lstm_dim': 128,                            # lstm layer dimension
        'use_gate': True,                           # use filter gate
        'gate_activation': 'sigmoid',               # gate activation function
        'crs_term': True,                           # add cross term to classifier
        'num_negatives': 5,                         # number of negative examples
        'optimization_method': 'adam',              # accepted value: 'sgd', 'adam'
        'adam_beta1': 0.9,                          # 1st adam hyperparameter
        'adam_beta2': 0.999,                        # 2nd adam hyperparameter
        'adam_epsilon': 1e-4,                       # 3rd adam hyperparameter

        'use_definitions': False,
    }

    #####################
    # launch experiment #
    #####################
    launch_exp(settings)
