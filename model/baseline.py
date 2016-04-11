'''
created on Apr 11, 2016

@author: leolong
'''

import os
import sys
import json
import random

_filtering = True
_threshold = 1000
random.seed(1234)

def most_frequent(path, split = 'test'):
    fnames = os.listdir(path)
    fnames.sort()
    random.shuffle(fnames)

    if split == 'train':
        partition = fnames[:95]
    elif split == 'valid':
        partition = fnames[95:102]
    else:
        partition = fnames[102:]

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

if __name__ == '__main__':
    most_frequent(sys.argv[1], split = 'valid')
