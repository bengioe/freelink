'''
created on Mar 6, 2016

@author: leolong
'''


import os
import sys
import json
import urllib


##########################
# fetch mid json payload #
##########################
def fetch_mid(guid, path):
    url = 'http://www.freebase.com/ajax/156b.lib.www.tags.svn.freebase-site.googlecode.dev/cuecard/mqlread.ajax?&query=%7B+%22lang%22%3A+%22%2Flang%2Fen%22%2C+%22query%22%3A+%5B%7B+%22guid%22%3A+%22%23{0}%22%2C+%22id%22%3A+null%2C+%22mid%22%3A+null%2C+%22name%22%3A+null%2C+%22type%22%3A+%5B%5D+%7D%5D+%7D'.format(guid)
    name = '{0}.json'.format(guid)
    urllib.urlretrieve(url, path + name)


#############################################
# detect mid payload neede to be re-fetched #
#############################################
def failed_mid(fnames, path):
    failed = []

    for name in fnames:
        guid = name[:len(name) - 5]
        data = None
        try:
            f = open(path + name, 'r')
            data = json.load(f)
            f.close()
        except:
            failed.append(guid)
            continue
        mid = None
        try:
            mid = data['result']['result'][0]['mid']
        except:
            failed.append(guid)
            continue

    return failed


##########################
# fetch lex json payload #
##########################
def fetch_lex(guid, mid, path):
    url = 'https://www.googleapis.com/freebase/v1/topic{0}?filter=/common/topic/description'.format(mid)
    name = '{0}.json'.format(guid)
    urllib.urlretrieve(url, path + name)


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

        count = 0
        for g in guid['guid']:
            count += 1
            print '- retriving guid {0} # {1}\n'.format(g, count)
            fetch_mid(g, midpath)

        print '# total mid: {0}'.format(count)

    #######################################
    # option: re-fetch failed mid payload #
    #######################################
    if sys.argv[1] == '-remid':
        fnames = os.listdir(midpath)
        fnames.sort()

        count = 0
        while True:
            failed = failed_mid(fnames, midpath)
            if len(failed) == 0:
                break

            count += 1
            print 'Iteration {0}'.format(count)
            for g in failed:
                print '- re-fetching {0}\n'.format(g)
                fetch_mid(g, midpath)

        print 'Re-fetching process complete, # iteration: {0}'.format(count)

    #######################
    # option: guid to mid #
    #######################
    if sys.argv[1] == '-guid2mid':
        fnames = os.listdir(midpath)
        fnames.sort()

        guid2mid = {}
        count = 0

        for name in fnames:
            f = open(midpath + name, 'r')
            data = json.load(f)
            f.close()

            guid = name[:len(name) - 5]
            mid = data['result']['result'][0]['mid']
            guid2mid[guid] = mid
            count += 1

        print '# mid json payload: {0}'.format(count)
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

        count = 0
        for key, value in guid2mid.iteritems():
            count += 1
            print '- retriving lex {0} # {1}\n'.format(value, count)
            fetch_lex(key, value, lexpath)

        print '# total lex: {0}'.format(count)

    #######################################
    # option: re-fetch failed lex payload #
    #######################################
    '''
    ......
    '''

    #######################
    # option: guid to lex #
    #######################
    '''
    ......
    '''
