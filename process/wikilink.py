'''
created on Feb 8, 2016

@author: leolong
'''


from ttypes import *
from constants import *
from gzip import GzipFile
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from newspaper import Article
from nltk.tokenize import word_tokenize
from nltk.tokenize import wordpunct_tokenize

import os
import re
import sys
import ftfy
import json
import time
import string
import logging
import jsonrpc


###############################################
# function to extract data from a single file #
###############################################
def extract_data(fname, server, loadp, savep):
    # load file using thrift frameword
    stream = GzipFile(loadp + fname)
    protocol = TBinaryProtocol.TBinaryProtocol(TTransport.TBufferedTransport(stream))
    # dataset
    data = {'data': []}
    count = 0

    while True:
        # read next data entry
        page = WikiLinkItem()
        try:
            page.read(protocol)
            count += 1
        except:
            stream.close()
            break

        # print progress
        print '- processing FILE {0} ENTRY # {1}'.format(fname, count)
        print '\t $ URL: {0}'.format(page.url)

        # ignore urls with 'ftp' prefix
        if page.url[:3] == 'ftp':
            print '\t\t ###### Ftp prefix detected in url (ignore) ###### \n'
            continue
        # ignore urls without 'html' suffix
        if page.url[len(page.url) - 4:] != 'html':
            print '\t\t ###### Non-html suffix detected in url (ignore) ###### \n'
            continue
        # ignore pages with empty dom
        if page.content.dom == None:
            print '\t\t ###### Empty dom detected (ignore) ###### \n'
            continue

        # logging basic info
        logging.info('')
        logging.info('FILE {0} # {1}'.format(fname, count))
        logging.info('url: {0}'.format(page.url))

        # extract all unique entities among mentions
        before = time.time()
        entities = extract_entities(page.mentions)
        # logging
        logging.info('\t * time spent on extract_entities: {0} s'.format(time.time() - before))

        # discard pages with single entity
        if len(entities) < 2:
            print '\t\t ###### Single entity found (discard) ###### \n'
            continue
        print '\t $ # Entities:', len(entities)

        # mark dom string
        before = time.time()
        html = mark_dom(page.content.dom, entities)
        # logging
        logging.info('\t * time spent on mark_dom: {0} s'.format(time.time() - before))

        # parse marked html
        before = time.time()
        news = Article(page.url, language = 'en')
        news.set_html(html)

        try:
            # attempt to parse marked html
            news.parse()
            # logging
            logging.info('\t * time spent on news.parse(): {0} s'.format(time.time() - before))
        except:
            print '\t\t ###### Parsing failed (discard) ###### \n'
            continue

        # tokenize text
        before = time.time()
        text = None
        try:
            text = ftfy.fix_text(news.text)
            text = text.encode('ascii', 'ignore')
            text = seperate_punc(word_tokenize(text))
            # logging
            logging.info('\t * time spent on tokenization: {0} s'.format(time.time() - before))
            '''
            # Unreliable
            text = ' '.join(word_tokenize(text))
            text = ' '.join(wordpunct_tokenize(text))
            '''
            '''
            # Un-comment to tokenize with CoreNLP Server
            payload = json.loads(server.parse(text))
            text = recover_text(payload)
            '''
        except:
            print '\t\t ###### Tokenization failed (discard) ###### \n'
            continue

        # save processed data
        print '\t $ Entry Saved \n'
        data['data'].append({'text': text, 'dict': entities})


    # Save data to json file
    print '****** {0}.json saved ******\n'.format(fname[:3])
    f = open(savepath + '{0}.json'.format(fname[:3]), 'w')
    json.dump(data, f, indent = 4)
    f.close()

'''
##############################################
# function to check potential non-html urls #
##############################################
def is_url_html(url):
    return (url[len(url) - 4:] == 'html') or (url[len(url) - 1] == '/')
'''
'''
####################################################
# function to check for undesired keywords in urls #
####################################################
def contain_keywords(url):
    url = url.lower()
    keywords = ['download', 'feed', 'file', 'pdf', 'doc', 'ppt']

    for word in keywords:
        if word in url:
            return True

    return False
'''

#######################################
# function to extract unique entities #
#######################################
def extract_entities(mentions):
    result = []
    seen = set([])

    for m in mentions:
        if m.freebase_id == None:
            continue
        if m.wiki_url == None:
            continue
        if m.wiki_url in seen:
            continue
        result.append(m)
        seen.add(m.wiki_url)

    dictionary = {}

    for i in range(0, len(result)):
        mark = 'marker_{0}'.format(i)
        dictionary[mark] = {'anchor_text': result[i].anchor_text,
                            'freebase_id': result[i].freebase_id,
                            'wiki_url':    result[i].wiki_url}

    # return type: entity dictionary
    return dictionary

###############################################
# function to mark all occurences of entities #
###############################################
def mark_dom(dom, entities):
    prevhtml = dom
    currhtml = ''

    for key, value in entities.iteritems():
        # find the positions of wiki_urls
        before = time.time()
        iterator, match = None, None
        try:
            iterator = re.finditer(value['wiki_url'], prevhtml)
            match = [m for m in iterator]
            # logging
            logging.info('\t\t * time spent on re.finditer(): {0} s'.format(time.time() - before))
        except:
            continue

        # fint positions of open and close tags
        before = time.time()
        tagidx = find_tagidx(prevhtml, match)
        # logging
        logging.info('\t\t * time spent on find_tagidx: {0} s'.format(time.time() - before))

        # Skip this entity if no match was found
        if len(tagidx) == 0:
            continue

        before = time.time()
        for i in range(0, len(tagidx)):
            # find index to isolate entity mention
            start, end = -1, -1
            if i == 0:
                start, end = 0, tagidx[i]['start']
            else:
                start, end = tagidx[i - 1]['end'], tagidx[i]['start']
            # substitude entity mention with its marker
            currhtml += prevhtml[start:end] + ' {0} '.format(key)
            # concatinate remaining file
            if i == len(tagidx) - 1:
                currhtml += prevhtml[tagidx[i]['end']:]

        prevhtml = currhtml
        currhtml = ''
        # logging
        logging.info('\t\t * time spent on re-constructing html: {0} s'.format(time.time() - before))

    # return type: marked string
    return prevhtml

###########################################
# function to find the index of html tags #
###########################################
def find_tagidx(html, match):
    tagidx = []

    for m in match:
        sidx, eidx = m.start(), m.end()

        count = 0
        while html[sidx:sidx + 2] != '<a':
            sidx -= 1
            count += 1
            if count >= 200:
                return []

        count = 0
        while html[eidx - 2:eidx] != 'a>':
            eidx += 1
            count += 1
            if count >= 200:
                return []

        tagidx.append({'start': sidx, 'end': eidx})

    # return type: tag index pair list
    return tagidx


#############################################
# function to handle punctuation at the end #
#############################################
def seperate_punc(tokens):
    words = []
    puncs = set(',.:;?!')

    for t in tokens:
        end = len(t) - 1
        if (t[end] not in puncs) or (t in puncs):
            words.append(t)
        else:
            words.append(t[:end])
            words.append(t[end])

    # return type: processed string
    return ' '.join(words)


##############################################
# function to recover text from json payload #
##############################################
def recover_text(payload):
    words = []

    for s in payload['sentences']:
        for w in s['words']:
            words.append(w[0])

    return ' '.join(words)


#################
# main function #
#################
if __name__ == '__main__':
    loadpath = '/scratch/data/wikilink/raw/'
    savepath = '/scratch/data/wikilink/ext/'

    if sys.argv[1] == 'extract':
        logging.basicConfig(filename = 'process.log', level = logging.INFO)
        fnames = os.listdir(loadpath)
        server = jsonrpc.ServerProxy(jsonrpc.JsonRpc20(),
                                     jsonrpc.TransportTcpIp(addr = ('127.0.0.1', 8080)))

        fnames.sort()
        for i in range(int(sys.argv[2]), int(sys.argv[3])):
            print '****** processing {0} ******\n'.format(fnames[i])
            extract_data(fnames[i], server, loadpath, savepath)
