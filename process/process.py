'''
created on Mar 12, 2016

@author: leolong
'''

import os
import sys
import json
import numpy
import string
import random
import cPickle
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

_filtering = True
_threshold = 1000
random.seed(1234)

def load_embedding(version):
    return cPickle.load(open('/scratch/data/embedding/' + 'glove.{0}.300d.pkl'.format(version), 'r'))

def name_embedding(name, embedding):
    vects = []
    for word in word_tokenize(name):
        if word in embedding['glove']:
            vects.append(embedding['glove'][word])
    if len(vects) > 0:
        return numpy.mean(vects, axis = 0)
    return embedding['mean']

def lex_embedding(sent, embedding):
    vects = []
    for word in sent.split():
        if word in embedding['glove']:
            vects.append(embedding['glove'][word])
    if len(vects) > 0:
        return numpy.mean(vects, axis = 0)
    return embedding['mean']

def flex_embedding(sent, embedding, remove):
    vects = []
    for word in sent.split():
        if (word in embedding['glove']) and (word not in remove):
            vects.append(embedding['glove'][word])
    if len(vects) > 0:
        return numpy.mean(vects, axis = 0)
    return embedding['mean']

def transform_train_doc(doc, num_negs, word2idx, voc_size, id2ee, guids):
    text, lex = [], []
    for word in doc['text'].lower().split():
        if word in doc['dict']:
            keyset = doc['dict'].keys()
            keyset.sort()
            keyset.remove(word)

            random.shuffle(keyset)
            sample = keyset[random.randint(0, len(keyset) - 1)]

            pos_id = doc['dict'][word]['freebase_id']
            pos_ee = id2ee[pos_id]

            neg_id = doc['dict'][sample]['freebase_id']
            neg_ee = id2ee[neg_id]

            text.append(voc_size + 1)
            if pos_id != neg_id:
                lex.append([pos_ee, neg_ee] + sample_negatives(num_negs - 1, pos_id, neg_id, id2ee, guids))
            else:
                lex.append([pos_ee] + sample_negatives(num_negs, pos_id, neg_id, id2ee, guids))
        elif word not in word2idx:
            text.append(voc_size)
        else:
            text.append(word2idx[word])
    return numpy.int32(text), lex

def sample_negatives(num_negs, pos_id, neg_id, id2ee, guids):
    neg_ees = []

    while num_negs > 0:
        sample = guids[random.randint(0, len(guids) - 1)]
        if (sample == pos_id) or (sample == neg_id):
            continue
        neg_ees.append(id2ee[sample])
        num_negs -= 1

    return neg_ees

def transform_test_doc(doc, word2idx, voc_size, id2ee):
    text, lex = [], []
    for word in doc['text'].lower().split():
        if word in doc['dict']:
            keyset = doc['dict'].keys()
            keyset.sort()
            keyset.remove(word)
            random.shuffle(keyset)

            pos_id = doc['dict'][word]['freebase_id']
            neg_ids = []
            for k in keyset:
                if doc['dict'][k]['freebase_id'] != pos_id:
                    neg_ids.append(doc['dict'][k]['freebase_id'])

            text.append(voc_size + 1)
            lex.append([id2ee[pos_id]] + [id2ee[i] for i in neg_ids])
        elif word not in word2idx:
            text.append(voc_size)
        else:
            text.append(word2idx[word])
    return numpy.int32(text), lex

if __name__ == '__main__':
    #####################################################################
    # produce the vocabulary by selecting the top k most frequent words #
    #####################################################################
    if sys.argv[1] == '-vocabulary':
        loadp = '/scratch/data/wikilink/ext/'
        savep = '/scratch/data/freelink/'

        # compute the word frequencies of the data set #
        w_counts = {}
        fnames = os.listdir(loadp)
        fnames.sort()

        for name in fnames:
            print '- processing file {0}'.format(name)
            data = json.load(open(loadp + name, 'r'))

            for doc in data['data']:
                if _filtering and (len(doc['text'].split()) > _threshold):
                    continue
                for word in doc['text'].lower().split():
                    if word in doc['dict']:
                        continue
                    if word not in w_counts:
                        w_counts[word] = 0
                    w_counts[word] += 1

        # sort words by their frequencies #
        w_sorted = sorted(w_counts.items(), key = lambda item : item[1], reverse = True)
        size = int(sys.argv[2])
        print 'Total # of words: {0}; selected vocabulary size: {1}'.format(len(w_sorted), size)

        # save word-to-index dictionary #
        word2idx = {}
        for i in range(0, size):
            word2idx[w_sorted[i][0]] = i

        json.dump(word2idx, open(savep + 'word2idx.json', 'w'), indent = 4)
        print '\t word2idx saved!'

    ################################
    # produce entity proxy vectors #
    ################################
    if sys.argv[1] == '-embedding':
        loadp = '/scratch/data/freebase/'
        savep = '/scratch/data/freelink/'
        embedding = load_embedding('840B')

        # generate embedding based on entity names #
        guid2name = json.load(open(loadp + 'guid2name.json', 'r'))
        ee_rand = {}
        ee_name = {}

        for guid, name in guid2name.iteritems():
            ee_rand[guid] = numpy.random.uniform(-1, 1, 300).astype('float32')
            ee_name[guid] = name_embedding(name, embedding).astype('float32')

        print 'Dumping random embeddings ......'
        cPickle.dump(ee_rand, open(savep + 'ee_rand.pkl', 'w'), - 1)
        print '\t Saved!'

        print 'Dumping name embeddings ......'
        cPickle.dump(ee_name, open(savep + 'ee_name.pkl', 'w'), -1)
        print '\t Saved!'

        # generate embedding based on lexical resources #
        guid2lex = json.load(open(loadp + 'guid2lex.json', 'r'))
        remove = set(stopwords.words('english')) | set(string.punctuation)
        ee_lex = {}
        ee_flex = {}

        for guid, lex in guid2lex.iteritems():
            sent = sent_tokenize(lex)[0]
            ee_lex[guid] = lex_embedding(sent, embedding).astype('float32')
            ee_flex[guid] = flex_embedding(sent, embedding, remove).astype('float32')

        print 'Dumping lexical embeddings ......'
        cPickle.dump(ee_lex, open(savep + 'ee_lex.pkl', 'w'), -1)
        print '\t Saved!'

        print 'Dumping filtered lexical embeddings ......'
        cPickle.dump(ee_flex, open(savep + 'ee_flex.pkl', 'w'), -1)
        print '\t Saved!'

    #####################################
    # produce train / valid / test data #
    #####################################
    if sys.argv[1] == '-data':
        loadp = '/scratch/data/wikilink/ext/'
        savep = '/scratch/data/freelink/'
        num_negs = 5

        fnames = os.listdir(loadp)
        fnames.sort()
        random.shuffle(fnames)

        partition = ['train', 'valid', 'test']
        files = [fnames[0:97], fnames[97:103], fnames[103:109]]

        word2idx = json.load(open(savep + 'word2idx.json', 'r'))
        voc_size = len(word2idx)

        for p, f in zip(partition, files):
            print 'Transforming {0} data......'.format(p)
            path = savep + '{0}/'.format(p)
            if not os.path.exists(path):
                os.makedirs(path)

            for version in ['rand', 'name', 'lex', 'flex']:
                id2ee = cPickle.load(open(savep + 'ee_{0}.pkl'.format(version), 'r'))
                guids = id2ee.keys()
                x, e = [], []

                for fname in f:
                    data = json.load(open(loadp + fname, 'r'))
                    for doc in data['data']:
                        if _filtering and (len(doc['text'].split()) > _threshold):
                            continue
                        if p == 'train':
                            text, lex = transform_train_doc(doc, num_negs, word2idx, voc_size, id2ee, guids)
                        else:
                            text, lex = transform_test_doc(doc, word2idx, voc_size, id2ee)
                        x.append(text)
                        e.append(lex)

                print '\t - version [{0}] ready! # negative examples: {1}'.format(version, num_negs)
                cPickle.dump({'x': x, 'e': e}, open(path + '{0}_{1}.pkl'.format(p, version), 'w'), -1)

    ###################################
    # count train / valid / test data #
    ###################################
    if sys.argv[1] == '-count':
        path = '/scratch/data/freelink/'
        train = cPickle.load(open(path + 'train/train_name.pkl', 'r'))
        valid = cPickle.load(open(path + 'valid/valid_name.pkl', 'r'))
        test = cPickle.load(open(path + 'test/test_name.pkl', 'r'))

        print '# docs available: {0}'.format(len(train['x']) + len(valid['x']) + len(test['x']))
        print '\t train: {0}; valid: {1}; test: {2}'.format(len(train['x']), len(valid['x']), len(test['x']))

        print '# empty slots: {0}'.format(sum([len(i) for i in train['e']]) +
                                          sum([len(i) for i in valid['e']]) +
                                          sum([len(i) for i in test['e']]))
        print '\t train: {0}; valid: {1}; test: {2}'.format(sum([len(i) for i in train['e']]),
                                                            sum([len(i) for i in valid['e']]),
                                                            sum([len(i) for i in test['e']]))
