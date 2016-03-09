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
    guidpath = '/scratch/data/freebase/'
    midpath = '/scratch/data/freebase/mid/'
    lexpath = '/scratch/data/freebase/lex/'

    ##############################
    # option: get mid using guid #
    ##############################
    if sys.argv[1] == '-mid':
        f = open(guidpath + 'guid.json', 'r')
        guid = json.load(f)
        f.close()

        succ = 0
        fail = 0
        count = 0

        for g in guid['guid']:
            url = 'http://www.freebase.com/ajax/156b.lib.www.tags.svn.freebase-site.googlecode.dev/cuecard/mqlread.ajax?&query=%7B+%22lang%22%3A+%22%2Flang%2Fen%22%2C+%22query%22%3A+%5B%7B+%22guid%22%3A+%22%23{0}%22%2C+%22id%22%3A+null%2C+%22mid%22%3A+null%2C+%22name%22%3A+null%2C+%22type%22%3A+%5B%5D+%7D%5D+%7D'.format(g)
            name = '{0}.json'.format(g)

            count += 1
            print '- retriving guid {0} # {1}'.format(g, count)

            try:
                urllib.urlretrieve(url, midpath + name)
                succ += 1
                print '\t query complete (saved)!\n'
            except:
                fail += 1
                print '\t query failed (discard)......\n'
                continue

        print '# total mid: {0}; complete: {1}, failed: {2}'.format(count, succ / float(count), fail / float(count))

    #######################
    # option: guid to mid #
    #######################
    if sys.argv[1] == '-guid2mid':
        fnames = os.listdir(midpath)
        fnames.sort()

        guid2mid = {}
        miss = 0
        count = 0

        for i in range(0, len(fnames)):
            f = open(midpath + fnames[i], 'r')
            data = json.load(f)
            f.close()

            guid = data['result']['result'][0]['guid'][1:]
            mid = data['result']['result'][0]['mid']
            if len(mid) == 0:
                miss += 1

            guid2mid[guid] = mid
            count += 1

        print '# mid-json: {0}; # missing mid: {1}'.format(count, miss)
        g = open(guidpath + 'guid2mid.json', 'w')
        json.dump(guid2mid, g, indent = 4)
        g.close()

    #############################
    # option: get lex using mid #
    #############################
    if sys.argv[1] == '-lex':
        f = open(guidpath + 'guid2mid.json', 'r')
        guid2mid = json.load(f)
        f.close()

        succ = 0
        fail = 0
        count = 0

        for key, value in guid2mid.iteritems():
            url = 'https://www.googleapis.com/freebase/v1/topic{0}?filter=/common/topic/description'.format(value)
            name = '{0}.json'.format(key)

            count += 1
            print '- retriving mid {0} # {1}'.format(key, count)

            try:
                urllib.urlretrieve(url, lexpath + name)
                succ += 1
                print '\t query complete (saved)!\n'
            except:
                fail += 1
                print '\t query failed (discard)......\n'
                continue

        print '# total lex: {0}; complete: {1}, failed: {2}'.format(count, succ / float(count), fail / float(count))

    #######################
    # option: guid to lex #
    #######################
    if sys.argv[1] == 'guid2lex':
        fnames = os.listdir(lexpath)
        fname.sort()

        guid2lex = {}
        miss = 0
        count = 0
