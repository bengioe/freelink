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
import string
import jsonrpc


###############################################
# function to extract data from a single file #
###############################################
def extract_data(fname, server, loadp, savep):
    # load file using thrift frameword
    stream = GzipFile(loadp + fname)
    protocol = TBinaryProtocol.TBinaryProtocol(TTransport.TBufferedTransport(stream))
    data = {'data': []}

    while True:
        # read next data entry
        page = WikiLinkItem()
        try:
            page.read(protocol)
        except:
            stream.close()
            break

        # print progress
        print '- processing'
        print '\t url:', page.url

        # ignore pages with 'ftp' urls
        if page.url[:3] == 'ftp':
            print '\t\t ###### Ftp url found (ignore) ###### \n'
            continue
        # ignore pages with undesired keywords in urls
        if contain_keywords(page.url):
            print '\t\t ###### Undesired keywords in url (ignore) ###### \n'
            continue
        # ignore pages with 'None' dom
        if page.content.dom == None:
            print '\t\t ###### None dom found (ignore) ###### \n'
            continue

        # extract all unique entities among mentions
        entities = extract_entities(page.mentions)
        # discard pages with single entity
        if len(entities) < 2:
            print '\t\t ###### Single entity found (discard) ###### \n'
            continue

        print '\t # entities:', len(entities)

        # mark dom string
        html = mark_dom(page.content.dom, entities)
        # discard pages that failed to be marked
        if html == None:
            print '\t\t ###### Alignment failed (discard) ###### \n'
            continue

        # parse marked html
        news = Article(page.url, language = 'en')
        news.set_html(html)
        # discard pages that cannot be parsed
        try:
            news.parse()
        except:
            print '\t\t ###### Parsing failed (discard) ###### \n'
            continue

        # tokenize text
        text = None
        try:
            text = ftfy.fix_text(news.text)
            text = text.encode('ascii', 'ignore')
            text = seperate_punc(word_tokenize(text))

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
        print '\t [DATA SAVED] \n'
        data['data'].append({'text': text, 'dict': entities})

    # Save data to json file
    print '****** {0}.json saved ******\n'.format(fname[:3])
    f = open(savepath + '{0}.json'.format(fname[:3]), 'w')
    json.dump(data, f, indent = 4)
    f.close()


####################################################
# function to check for undesired keywords in urls #
####################################################
def contain_keywords(url):
    url = url.lower()
    keywords = ['download', 'file', 'pdf', 'doc', 'ppt']

    for word in keywords:
        if word in url:
            return True

    return False


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
        if m.freebase_id in seen:
            continue
        result.append(m)
        seen.add(m.freebase_id)

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
        iterator, match = None, None
        try:
            iterator = re.finditer(value['wiki_url'], prevhtml)
            match = [m for m in iterator]
        except:
            # return type: 'None' to signal failure
            return None

        if len(match) == 0:
            continue

        tagidx = find_tagidx(prevhtml, match)

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

    # return type: marked string
    return prevhtml

###########################################
# function to find the index of html tags #
###########################################
def find_tagidx(html, match):
    index = []

    for m in match:
        sidx, eidx = m.start(), m.end()
        while html[sidx:sidx + 2] != '<a':
            sidx -= 1
        while html[eidx - 2:eidx] != 'a>':
            eidx += 1
        index.append({'start': sidx, 'end': eidx})

    # return type: tag index pair list
    return index


#############################################
# function to handle punctuation at the end #
#############################################
def seperate_punc(tokens):
    words = []
    puncs = set(',.:;!?')

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
        fnames = os.listdir(loadpath)
        server = jsonrpc.ServerProxy(jsonrpc.JsonRpc20(),
                                     jsonrpc.TransportTcpIp(addr = ('127.0.0.1', 8080)))

        fnames.sort()
        for i in range(0, len(fnames)):
            print '****** processing {0} ******\n'.format(fnames[i])
            extract_data(fnames[i], server, loadpath, savepath)
