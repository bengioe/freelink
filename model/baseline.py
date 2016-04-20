'''
created on Apr 11, 2016

@author: leolong
'''

import os
import sys
import json
import numpy
import random
import cPickle
from nltk.tokenize import sent_tokenize
from scipy.spatial.distance import cosine

_filtering = True
_threshold = 1000
random.seed(1234)

def most_frequent(path, split = 'test'):
    fnames = os.listdir(path)
    fnames.sort()
    random.shuffle(fnames)

    if split == 'train':
        partition = fnames[0:97]
    elif split == 'valid':
        partition = fnames[97:103]
    else:
        partition = fnames[103:109]

    correct = 0
    nblanks = 0

    for name in partition:
        data = json.load(open(path + name, 'r'))
        for doc in data['data']:
            if _filtering and (len(doc['text'].split()) > _threshold):
                continue
            pred = most_common_ent(doc)
            for word in doc['text'].lower().split():
                if word not in doc['dict']:
                    continue
                if word == pred:
                    correct += 1
                nblanks += 1

    print 'Most frequent baseline'
    print '\t - [correct]: {0}; [total]: {1}'.format(correct, nblanks)
    print '\t - acc. on {0} set: {1}'.format(split, correct / float(nblanks))

def most_common_ent(doc):
    ent2freq = {}
    for word in doc['text'].lower().split():
        if word not in doc['dict']:
            continue
        if word not in ent2freq:
            ent2freq[word] = 0
        ent2freq[word] += 1
    freq_sorted = sorted(ent2freq.items(), key = lambda item : item[1], reverse = True)
    return freq_sorted[0][0]

def cosine_similarity(d_path, e_path, split = 'test', version = 'flex'):
    fnames = os.listdir(d_path)
    fnames.sort()
    random.shuffle(fnames)

    if split == 'train':
        partition = fnames[0:97]
    elif split == 'valid':
        partition = fnames[97:103]
    else:
        partition = fnames[103:109]

    embeddings = cPickle.load(open(e_path + 'glove.840B.300d.pkl', 'r'))
    id2ee = cPickle.load(open(e_path + 'ee_{0}.pkl'.format(version), 'r'))

    correct = 0
    nblanks = 0

    for name in partition:
        data = json.load(open(d_path + name, 'r'))
        for doc in data['data']:
            if _filtering and (len(doc['text'].split()) > _threshold):
                continue
            mark2ee = get_marker_ees(doc['dict'], id2ee)
            for sent in sent_tokenize(doc['text']):
                if not contain_mark(sent, doc['dict']):
                    continue
                sent_emb = get_sent_emb(sent, embeddings)
                pred = least_cos_dist(sent_emb, mark2ee)
                for word in sent.lower().split():
                    if word not in doc['dict']:
                        continue
                    if word == pred:
                        correct += 1
                    nblanks += 1

    print 'Cosine similarity baseline'
    print '\t - [correct]: {0}; [total]: {1}'.format(correct, nblanks)
    print '\t - acc. on {0} set: {1}'.format(split, correct / float(nblanks))

def get_marker_ees(mark_dict, id2ee):
    mark2ee = {}
    for key, value in mark_dict.iteritems():
        mark2ee[key] = id2ee[value['freebase_id']]
    return mark2ee

def contain_mark(sent, mark_dict):
    for word in sent.lower().split():
        if word in mark_dict:
            return True
    return False

def get_sent_emb(sent, embeddings):
    vects = []
    for word in sent.lower().split():
        if word in embeddings['glove']:
            vects.append(embeddings['glove'][word])
    if len(vects) > 0:
        return numpy.mean(vects, axis = 0)
    return embeddings['mean']

def least_cos_dist(sent_emb, mark2ee):
    pred = 'marker_0'
    dist = 1.
    for key, value in mark2ee.iteritems():
        if cosine(sent_emb, value) < dist:
            pred = key
            dist = cosine(sent_emb, value)
    return pred

if __name__ == '__main__':
    # Most common entity baseline #
    if sys.argv[1] == '-freq':
        most_frequent('/scratch/data/wikilink/ext/', split = 'train')

    # Cosine similarity baseline #
    if sys.argv[1] == '-cosine':
        cosine_similarity('/scratch/data/wikilink/ext/', '/scratch/data/freelink/', split = 'train', version = 'flex')
