'''
created on Mar 12, 2016

@author leolong
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

def load_embedding(version):
    return cPickle.load(open('/scratch/data/embedding/' + 'glove.{0}.300d.pkl'.format(version), 'r'))

def name_embedding(name, embedding):
    vects = []
    for word in word_tokenize(name):
        if word in embedding['glove']:
            vects.append(embedding['glove'][word])
    if len(vects) > 0:
        return numpy.mean(vects, axis = 0, dtype = 'float32')
    return embedding['mean']

def lex_embedding(sent, embedding):
    vects = []
    for word in sent.split():
        if word in embedding['glove']:
            vects.append(embedding['glove'][word])
    if len(vects) > 0:
        return numpy.mean(vects, axis = 0, dtype = 'float32')
    return embedding['mean']

def flex_embedding(sent, embedding, remove):
    vects = []
    for word in sent.split():
        if (word in embedding['glove']) and (word not in remove):
            vects.append(embedding['glove'][word])
    if len(vects) > 0:
        return numpy.mean(vects, axis = 0, dtype = 'float32')
    return embedding['mean']

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

            for sample in data['data']:
                text = sample['text'].lower()
                for word in text.split():
                    if word not in w_counts:
                        w_counts[word] = 0
                    w_counts[word] += 1

        # sort words by their frequencies #
        w_sorted = sorted(w_counts.items(), key = lambda item : item[1], reverse = True)
        size = int(sys.argv[2])
        print 'Total # of words: {0}; selected vocabulary size: {1}'.format(len(w_sorted), size)

        # save word-to-index dictionary #
        vocabulary = {}
        for i in range(0, size):
            vocabulary[w_sorted[i][0]] = i

        json.dump(vocabulary, open(savep + 'vocabulary.json', 'w'), indent = 4)
        print '\t vocabulary saved!'

    ################################
    # produce entity proxy vectors #
    ################################
    if sys.argv[1] == '-embedding':
        loadp = '/scratch/data/freebase/'
        savep = '/scratch/data/freelink/'
        embedding = load_embedding('840B')

        # generate embedding based on entity names #
        guid2name = json.load(open(loadp + 'guid2name.json', 'r'))
        name_vects = {}

        for guid, name in guid2name.iteritems():
            name_vects[guid] = name_embedding(name, embedding)

        print 'Dumping name embeddings ...'
        cPickle.dump(name_vects, open(savep + 'name_vects.pkl', 'w'), -1)
        print '\t Saved'

        # generate embedding based on lexical resources #
        guid2lex = json.load(open(loadp + 'guid2lex.json', 'r'))
        remove = set(stopwords.words('english')) | set(string.punctuation)
        lex_vects = {}
        flex_vects = {}

        for guid, lex in guid2lex.iteritems():
            sent = sent_tokenize(lex)[0]
            lex_vects[guid] = lex_embedding(sent, embedding)
            flex_vects[guid] = flex_embedding(sent, embedding, remove)

        print 'Dumping lexical embeddings ...'
        cPickle.dump(lex_vects, open(savep + 'lex_vects.pkl', 'w'), -1)
        print '\t Saved'

        print 'Dumping filtered lexical embeddings ...'
        cPickle.dump(flex_vects, open(savep + 'flex_vects.pkl', 'w'), -1)
        print '\t Saved'

    #####################################
    # produce train / valid / test data #
    #####################################
    if sys.argv[1] == '-data':
        random.seed(1234)
        path = '/scratch/data/freelink/'
        word2idx = json.load(open(path + 'vocabulary.json', 'r'))
        vects = cPickle.load(open(path + '{0}_vects.pkl'.format(sys.argv[2]), 'r'))
        size = len(word2idx)

        for f in ['train', 'valid', 'test']:
            loadp = path + '{0}/'.format(f)
            fnames = os.listdir(loadp)
            fnames.sort()

            x, x_p = [], []
            for name in fnames:
                if name[-5:] != '.json':
                    continue

                data = json.load(open(loadp + name, 'r'))
                for doc in data['data']:
                    if len(doc['text'].split()) > int(sys.argv[2]):
                        continue

                    text, lex = [], []
                    for word in doc['text'].lower().split():
                        if word in doc['dict']:
                            text.append(size + 1)
                            pos = vects[doc['dict'][word]['freebase_id']]
                            entset = doc['dict'].keys()
                            entset.remove(word)
                            random.shuffle(entset)
                            sample = entset[random.randint(0, len(entset) - 1)]
                            neg = vects[doc['dict'][sample]['freebase_id']]
                            lex.append([pos, neg])
                        elif word not in word2idx:
                            text.append(size)
                        else:
                            text.append(word2idx[word])
                    x.append(numpy.int32(text))
                    x_p.append(lex)

            savep = loadp + '{0}/'.format(sys.argv[2])
            if not os.path.exists(savep):
                os.makedirs(savep)
            print 'Saving {0}.pkl ......'.format(f)
            cPickle.dump({'x': x, 'e': x_p}, open(savep + '{0}.pkl'.format(f), 'w'), -1)
