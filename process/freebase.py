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


##############################
# detect missing mid payload #
##############################
def miss_mid(fnames, guid):
    exist = set([])
    for name in fnames:
        exist.add(name[:len(name) - 5])

    miss = []
    for g in guid['guid']:
        if g not in exist:
            miss.append(g)

    return miss


#####################################################
# detect failed mid payload needed to be re-fetched #
#####################################################
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
def fetch_lex(guid, params, path):
    url = 'https://www.googleapis.com/freebase/v1/search?' + urllib.urlencode(params)
    name = '{0}.json'.format(guid)
    urllib.urlretrieve(url, path + name)
    # re-dump to format json payload #
    data = json.load(open(path + name, 'r'))
    json.dump(data, open(path + name, 'w'), indent = 4)


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
        succ = 0
        fail = 0
        for g in guid['guid']:
            count += 1
            print '- retriving guid {0} # {1}'.format(g, count)
            try:
                fetch_mid(g, midpath)
                succ += 1
                print '\t query complete!\n'
            except:
                fail += 1
                print '\t query failed......\n'

        print '# total mid: {0}; complete / failed: {1} / {2}'.format(count, succ, fail)

    #######################################
    # option: re-fetch failed mid payload #
    #######################################
    if sys.argv[1] == '-remid':
        #############################
        # fetch missing mid payload #
        #############################
        f = open(guidpath + 'guid.json', 'r')
        guid = json.load(f)
        f.close()

        i = 0
        while True:
            fnames = os.listdir(midpath)
            miss = miss_mid(fnames, guid)
            if len(miss) == 0:
                break
            i += 1
            print 'Miss iteration {0}'.format(i)
            for g in miss:
                print '\t - fetching {0}\n'.format(g)
                try:
                    fetch_mid(g, midpath)
                except:
                    continue
        print 'Fetching missing mid complete, # iteration cost: {0}\n'.format(i)

        ##################################
        # re-fetching failed mid payload #
        ##################################
        fnames = os.listdir(midpath)
        fnames.sort()

        j = 0
        while True:
            failed = failed_mid(fnames, midpath)
            if len(failed) == 0:
                break
            j += 1
            print 'Fail iteration {0}'.format(j)
            for g in failed:
                print '\t - re-fetching {0}\n'.format(g)
                try:
                    fetch_mid(g, midpath)
                except:
                    continue
        print 'Re-fetching failed mid complete, # iteration cost: {0}\n'.format(j)

        print 'Mid refill complete, cost {0} miss iterations / {1} fail iterations'.format(i, j)

    #######################
    # option: guid to mid #
    #######################
    if sys.argv[1] == '-guid2mid':
        fnames = os.listdir(midpath)
        fnames.sort()

        guid2mid = {}
        count = 0
        miss = 0

        for name in fnames:
            f = open(midpath + name, 'r')
            data = json.load(f)
            f.close()

            guid = name[:len(name) - 5]
            mid = data['result']['result'][0]['mid']
            guid2mid[guid] = mid
            count += 1
            if len(mid) == 0:
                miss += 1

        print '# mid json payload: {0}; missing {1}'.format(count, miss)
        g = open(guidpath + 'guid2mid.json', 'w')
        json.dump(guid2mid, g, indent = 4)
        g.close()

    #############################
    # option: get lex using mid #
    #############################
    if sys.argv[1] == '-lex':
        guid = json.load(open(guidpath + 'guid.json', 'r'))
        guid2mid = json.load(open(guidpath + 'guid2mid.json', 'r'))

        api_key = open('/home/2014/tlong22/.freebase_api_key').read()
        params = {
            'key': api_key,
            'output': '(description)',
            'query': None
        }

        count = 0
        succ = 0
        fail = 0
        for g in guid['guid'][0:99800]:
            count += 1
            print '- retriving lex {0} # {1}'.format(guid2mid[g], count)
            params['query'] = guid2mid[g]
            try:
                fetch_lex(g, params, lexpath)
                succ += 1
                print '\t query complete!\n'
            except:
                fail += 1
                print '\t query failed......\n'

        print '# total lex: {0}; complete / failed: {1} / {2}'.format(count, succ, fail)

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
