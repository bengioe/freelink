'''
created on Mar 6, 2016

@author: leolong
'''


import os
import sys
import json
import urllib
import wikilink
from nltk.tokenize import word_tokenize


##########################
# fetch mid json payload #
##########################
def fetch_mid(guid, path):
    url = 'http://www.freebase.com/ajax/156b.lib.www.tags.svn.freebase-site.googlecode.dev/cuecard/mqlread.ajax?&query=%7B+%22lang%22%3A+%22%2Flang%2Fen%22%2C+%22query%22%3A+%5B%7B+%22guid%22%3A+%22%23{0}%22%2C+%22id%22%3A+null%2C+%22mid%22%3A+null%2C+%22name%22%3A+null%2C+%22type%22%3A+%5B%5D+%7D%5D+%7D'.format(guid)
    name = '{0}.json'.format(guid)
    try:
        urllib.urlretrieve(url, path + name)
        # re-dump to format json payload #
        data = json.load(open(path + name, 'r'))
        json.dump(data, open(path + name, 'w'), indent = 4)
        print '\t query complete!\n'
    except:
        print '\t query failed......\n'
        return


##########################
# find missing mid files #
##########################
def miss_mid(fnames, guid):
    return miss_file(fnames, guid)


###########################
# find failed mid payload #
###########################
def failed_mid(fnames, path):
    return failed_json(fnames, path)


##########################
# fetch lex json payload #
##########################
def fetch_lex(guid, params, path):
    url = 'https://www.googleapis.com/freebase/v1/search?' + urllib.urlencode(params)
    name = '{0}.json'.format(guid)
    try:
        urllib.urlretrieve(url, path + name)
        # re-dump to format json payload #
        data = json.load(open(path + name, 'r'))
        json.dump(data, open(path + name, 'w'), indent = 4)
        print '\t query complete!\n'
    except:
        print '\t query failed......\n'
        return


##########################
# find missing lex files #
##########################
def miss_lex(fnames, guid):
    return miss_file(fnames, guid)


###########################
# find failed lex payload #
###########################
def failed_lex(fnames, path):
    return failed_json(fnames, path)


#############################
# detect missing file names #
#############################
def miss_file(fnames, guid):
    exist = set([])
    for name in fnames:
        exist.add(name[:len(name) - 5])

    miss = []
    for g in guid['guid']:
        if g not in exist:
            miss.append(g)

    return miss


######################################################
# detect failed json payload needed to be re-fetched #
######################################################
def failed_json(fnames, path):
    failed = []

    for name in fnames:
        guid = name[:len(name) - 5]
        data = None
        try:
            data = json.load(open(path + name, 'r'))
        except:
            failed.append(guid)
            continue
        try:
            status = data['status']
            if status == '200 OK':
                continue
            else:
                failed.append(guid)
        except:
            failed.append(guid)

    return failed


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
        guid = json.load(open(guidpath + 'guid.json', 'r'))

        count = 0
        for g in guid['guid']:
            count += 1
            print '- retriving guid {0} # {1}'.format(g, count)
            fetch_mid(g, midpath)

        print 'DONE! # total mid: {0}'.format(count)

    #######################################
    # option: re-fetch failed mid payload #
    #######################################
    if sys.argv[1] == '-remid':
        #############################
        # fetch missing mid payload #
        #############################
        guid = json.load(open(guidpath + 'guid.json', 'r'))

        i = 0
        while True:
            fnames = os.listdir(midpath)
            miss = miss_mid(fnames, guid)
            if len(miss) == 0:
                break
            i += 1

            print 'Miss iteration {0}'.format(i)
            for g in miss:
                print '- fetching {0}'.format(g)
                fetch_mid(g, midpath)

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
                print '- re-fetching {0}'.format(g)
                fetch_mid(g, midpath)

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
            data = json.load(open(midpath + name, 'r'))
            guid = name[:len(name) - 5]
            guid2mid[guid] = data['result']['result'][0]['mid']
            count += 1
            if len(guid2mid[guid]) == 0:
                miss += 1

        print '# mid json payload: {0}; missing {1}'.format(count, miss)
        print '\t guid2mid.json saved'
        json.dump(guid2mid, open(guidpath + 'guid2mid.json', 'w'), indent = 4)

    #############################
    # option: get lex using mid #
    #############################
    if sys.argv[1] == '-lex':
        api_key = open('/home/2014/tlong22/.freebase_api_key').read()
        params = {
            'key': api_key,
            'output': '(description)',
            'query': None
        }

        guid = json.load(open(guidpath + 'guid.json', 'r'))
        guid2mid = json.load(open(guidpath + 'guid2mid.json', 'r'))

        count = 0
        for g in guid['guid']:
            count += 1
            print '- retriving lex {0} # {1}'.format(guid2mid[g], count)
            params['query'] = guid2mid[g]
            fetch_lex(g, params, lexpath)

        print 'DONE! # total lex: {0}'.format(count)

    #######################################
    # option: re-fetch failed lex payload #
    #######################################
    if sys.argv[1] == '-relex':
        api_key = open('/home/2014/tlong22/.freebase_api_key').read()
        params = {
            'key': api_key,
            'output': '(description)',
            'query': None
        }
        ################################
        # fetching missing lex payload #
        ################################
        guid = json.load(open(guidpath + 'guid.json', 'r'))
        guid2mid = json.load(open(guidpath + 'guid2mid.json', 'r'))

        i = 0
        while True:
            fnames = os.listdir(lexpath)
            miss = miss_lex(fnames, guid)
            if len(miss) == 0:
                break
            i += 1

            print 'Miss iteration {0}'.format(i)
            for g in miss:
                print '- fetching {0}'.format(g)
                params['query'] = guid2mid[g]
                fetch_lex(g, params, lexpath)

        print 'Fetching missing lex complete, # iteration cost: {0}\n'.format(i)

        ##################################
        # re-fetching failed lex payload #
        ##################################
        fnames = os.listdir(lexpath)
        fnames.sort()

        j = 0
        while True:
            failed = failed_lex(fnames, lexpath)
            if len(failed) == 0:
                break
            j += 1

            print 'Fail iteration {0}'.format(j)
            for g in failed:
                print '- re-fetching {0}'.format(g)
                params['query'] = guid2mid[g]
                fetch_lex(g, params, lexpath)

        print 'Re-fetching failed lex complete, # iteration cost: {0}\n'.format(j)
        print 'Lex refill complete, cost {0} miss iterations / {1} fail iterations'.format(i, j)

    #######################
    # option: guid to lex #
    #######################
    if sys.argv[1] == '-guid2lex':
        fnames = os.listdir(lexpath)
        fnames.sort()

        guid2lex = {}
        count = 0
        miss = 0

        for name in fnames:
            data = json.load(open(lexpath + name, 'r'))
            guid = name[:len(name) - 5]
            guid2lex[guid] = seperate_delimiter(word_tokenize(data['result'][0]['output']['description']['/common/topic/description'][0]))
            count += 1
            if len(guid2lex[guid]) == 0:
                miss += 1

        print '# lex json payload: {0}; missing {1}'.format(count, miss)
        print '\t guid2lex.json saved'
        json.dump(guid2lex, open(guidpath + 'guid2lex.json', 'w'), indent = 4)
