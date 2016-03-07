'''
created on Mar 6, 2016

@author: leolong
'''


import os
import sys
import json
import urllib


#################
# main function #
#################
if __name__ == '__main__':
    datapath = '/scratch/data/'
    midpath = '/scratch/data/freebase/mid/'
    lexpath = '/scratch/data/freebase/lex/'

    ##############################
    # option: get mid using guid #
    ##############################
    if sys.argv[1] == '-mid':
        f = open(datapath + 'guids.json')
        guids = json.load(f)
        f.close()

        count = 0

        for g in guids['guid']:
            url = 'http://www.freebase.com/ajax/156b.lib.www.tags.svn.freebase-site.googlecode.dev/cuecard/mqlread.ajax?&query=%7B+%22lang%22%3A+%22%2Flang%2Fen%22%2C+%22query%22%3A+%5B%7B+%22guid%22%3A+%22%23{0}%22%2C+%22id%22%3A+null%2C+%22mid%22%3A+null%2C+%22name%22%3A+null%2C+%22type%22%3A+%5B%5D+%7D%5D+%7D'.format(g)
            name = '{0}.json'.format(e)
            count += 1

            print '- retriving guid {0} # {1}'.format(e, count)

            try:
                urllib.urlretrieve(url, midpath + name)
                print '\t query complete (saved)!\n'
            except:
                print '\t query failed (discard)......\n'
                continue
