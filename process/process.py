'''
created on Mar 12, 2016

@author leolong
'''

import os
import sys
import json

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
