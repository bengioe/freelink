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
from nltk import word_tokenize
import re
import sys
import json


###############################################
# function to extract data from a single file #
###############################################
def extract_data(fname, loadp, savep):
    # load file using thrift frameword
    stream = GzipFile(loadp + fname)
    protocol = TBinaryProtocol.TBinaryProtocol(TTransport.TBufferedTransport(stream))
    data = {'data': []}

    for i in range(0, 30):
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
        # ignore pages with 'None' dom
        if page.content.dom == None:
            print '\t\t ###### None dom found (ignore) ###### \n'
            continue

        # extract all unique entities among mentions
        entities = build_dict(extract_entities(page.mentions))
        # ignore pages with single entity
        if len(entities) < 2:
            print '\t\t ###### Single entity found (ignore) ###### \n'
            continue

        # mark dom string
        html = mark_dom(page.content.dom, entities)
        # skip pages that failed to be marked
        if html == None:
            print '\t\t ###### Alignment failed (discard) ###### \n'
            continue

        news = Article(page.url, language = 'en')
        news.set_html(page.content.dom)
        f = open(savep + '{0}.html'.format(i), 'w')
        f.write(news.html.encode('ascii', 'ignore'))
        f.close()
        news.set_html(html)
        f = open(savep + '{0}_marked.html'.format(i), 'w')
        f.write(news.html.encode('ascii', 'ignore'))
        f.close()

        g = open(savep + '{0}.json'.format(i), 'w')
        json.dump(entities, g, indent = 4)
        g.close()


#######################################
# function to extract unique entities #
#######################################
def extract_entities(mentions):
    result = []
    seen = set([])

    for m in mentions:
        if m.freebase_id == None:
            continue
        if m.freebase_id in seen:
            continue
        result.append(m)
        seen.add(m.freebase_id)

    # return type: mention-object list
    return result


#######################################
# function to build entity dictionary #
#######################################
def build_dict(entities):
    dictionary = {}

    for i in range(0, len(entities)):
        mark = 'marker_{0}'.format(i)
        dictionary[mark] = {'anchor_text': entities[i].anchor_text,
                            'freebase_id': entities[i].freebase_id,
                            'wiki_url': entities[i].wiki_url}

    # return type: entity dictionary (key: marker token)
    return dictionary

###############################################
# function to mark all occurences of entities #
###############################################
def mark_dom(dom, entities):
    prevhtml = dom
    currhtml = ''

    for key, value in entities.iteritems():
        iterator = None
        match = None
        try:
            iterator = re.finditer(value['wiki_url'], prevhtml)
            match = [m for m in iterator]
        except:
            # return type: 'None' to signal failure
            return None

        index = find_tagidx(prevhtml, match)
        for i in range(0, len(index)):
            # isolate entity mention
            start1, end1, start2, end2 = -1, -1, -1, -1
            if len(index) == 1:
                start1, end1, start2, end2 = 0, index[i][0], index[i][1], len(prevhtml)
            elif i == 0:
                start1, end1, start2, end2 = 0, index[i][0], index[i][1], index[i + 1][0]
            elif i == len(index) - 1:
                start1, end1, start2, end2 = index[i - 1][1], index[i][0], index[i][1], len(prevhtml)
            else:
                start1, end1, start2, end2 = index[i - 1][1], index[i][0], index[i][1], index[i + 1][0]
            # mark entity mention
            currhtml += prevhtml[start1:end1] + key + prevhtml[start2:end2]

        prevhtml = currhtml
        currhtml = ''

    # return type: marked string
    return prevhtml


# function to find the index of html tags
def find_tagidx(html, match):
    index = []

    for m in match:
        sidx = m.start()
        eidx = m.end()
        while html[sidx:sidx + 2] != '<a':
            sidx -= 1
        while html[eidx - 2:eidx] != 'a>':
            eidx += 1
        index.append((sidx, eidx))

    return index


# main function
if __name__ == '__main__':
    loadpath = '/scratch/data/wikilinks/raw/'
    savepath = '/scratch/data/wikilinks/'

    extract_data('001.gz', loadpath, savepath)
