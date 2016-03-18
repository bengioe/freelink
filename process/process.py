'''
created on Mar 12, 2016

@author leolong
'''

import os
import sys
import json
import numpy
import cPickle
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from string import punctuation

def load_embedding(version):
    return cPickle.load(open('/scratch/data/embedding/' + 'glove.{0}.300d.pkl'.format(version), 'r'))

if __name__ == '__main__':
    #####################################################################
    # produce the vocabulary by selecting the top k most frequent words #
    #####################################################################
    if sys.argv[1] == '-vocabulary':
        loadp = '/scratch/data/wikilink/ext/'
        savep = '/scratch/data/freebase/'

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
        path = '/scratch/data/freebase/'
        embedding = load_embedding('840B')

        # generate embedding based on entity names #
        guid2name = json.load(open(path + 'guid2name.json', 'r'))
        name_vects = {}

        for guid, name in guid2name.iteritems():
            vects = []
            for word in word_tokenize(name):
                if word in embedding['glove']:
                    vects.append(embedding['glove'][word])
            if len(vects) > 0:
                name_vects[guid] = numpy.mean(vects, axis = 0, dtype = 'float32')
            else:
                name_vects[guid] = embedding['mean']

        print 'Dumping name embeddings ...'
        cPickle.dump(name_vects, open(path + 'name_vects.pkl', 'w'), -1)
        print '\t Saved'

        # generate embedding based on lexical resources #
        guid2lex = json.load(open(path + 'guid2lex.json', 'r'))
        remove = set(stopwords.words('english')) | set(punctuation)
        lex_vects = {}
        flex_vects = {}

        for guid, lex in guid2lex.iteritems():
            vects_1 = []
            vects_2 = []
            for word in sent_tokenize(lex)[0].split():
                if word in embedding['glove']:
                    vects_1.append(embedding['glove'][word])
                if (word in embedding['glove']) and (word not in remove):
                    vects_2.append(embedding['glove'][word])
            lex_vects[guid] = numpy.mean(vects_1, axis = 0, dtype = 'float32')
            flex_vects[guid] = numpy.mean(vects_2, axis = 0, dtype = 'float32')

        print 'Dumping lexical embeddings ...'
        cPickle.dump(lex_vects, open(path + 'lex_vects.pkl', 'w'), -1)
        print '\t Saved'
        print 'Dumping filtered lexical embeddings ...'
        cPickle.dump(flex_vects, open(path + 'flex_vects.pkl', 'w'), -1)
        print '\t Saved'
