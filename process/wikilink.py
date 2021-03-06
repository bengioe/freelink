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

import os
import re
import sys
import ftfy
import json

_filtering = True
_threshold = 1000

###############################################
# function to extract data from a single file #
###############################################
def extract_data(fname, loadp, savep):
    ######################
    # initialize process #
    ######################
    stream = GzipFile(loadp + fname)
    protocol = TBinaryProtocol.TBinaryProtocol(TTransport.TBufferedTransport(stream))

    data = {'data': []}
    count = 0

    ####################
    # begin extraction #
    ####################
    while True:
        page = WikiLinkItem()
        try:
            page.read(protocol)
            count += 1
        except:
            stream.close()
            break

        print '- processing FILE {0} ENTRY # {1}'.format(fname, count)
        print '\t $ URL: {0}'.format(page.url)

        #####################
        # initial filtering #
        #####################
        if page.url[:3] == 'ftp':
            print '\t\t ###### Ftp prefix detected (ignore) ###### \n'
            continue
        if page.url[len(page.url) - 4:] != 'html':
            print '\t\t ###### Non-html suffix detected (ignore) ###### \n'
            continue
        if page.content.dom == None:
            print '\t\t ###### Empty dom detected (ignore) ###### \n'
            continue

        #######################
        # secondary filtering #
        #######################
        entities = extract_entities(page.mentions)
        if len(entities) < 2:
            print '\t\t ###### Single entity found (discard) ###### \n'
            continue

        print '\t $ # Entities:', len(entities)

        #########################
        # alignment and parsing #
        #########################
        html = mark_dom(page.content.dom, entities)

        news = Article(page.url, language = 'en')
        try:
            news.set_html(html)
            news.parse()
        except:
            print '\t\t ###### Parsing failed (discard) ###### \n'
            continue

        ################
        # tokenization #
        ################
        text = None
        try:
            text = ftfy.fix_text(news.text)
            text = text.encode('ascii', 'ignore')
            text = seperate_delimiter(word_tokenize(text))
        except:
            print '\t\t ###### Tokenization failed (discard) ###### \n'
            continue

        #######################
        # save processed data #
        #######################
        print '\t $ Entry # {0} Saved \n'.format(count)
        data['data'].append({'text': text, 'dict': entities})

    #####################
    # save as json file #
    #####################
    print '****** {0}.json saved ******\n'.format(fname[:3])
    f = open(savep + '{0}.json'.format(fname[:3]), 'w')
    json.dump(data, f, indent = 4)
    f.close()


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
    prev = dom
    curr = ''

    for key, value in entities.iteritems():
        ##########################
        # locate entity mentions #
        ##########################
        iterator = None
        match = None
        try:
            iterator = re.finditer(value['wiki_url'], prev)
            match = [m for m in iterator]
        except:
            continue

        tagidx = find_tagidx(prev, match)
        if len(tagidx) == 0:
            continue

        #####################
        # perform alignment #
        #####################
        for i in range(0, len(tagidx)):
            end = 0
            if i > 0:
                end = tagidx[i - 1]['end']

            curr += prev[end:tagidx[i]['start']] + ' {0} '.format(key)
            if i == len(tagidx) - 1:
                curr += prev[tagidx[i]['end']:]

        prev = curr
        curr = ''

    # return type: marked string
    return prev

###########################################
# function to find the index of html tags #
###########################################
def find_tagidx(html, match):
    tagidx = []

    for m in match:
        sidx, eidx = m.start(), m.end()
        ##############################
        # find the index of open tag #
        ##############################
        count = 0
        while html[sidx:sidx + 2] != '<a':
            sidx -= 1
            count += 1
            if count >= 100:
                return []
        ###############################
        # find the index of close tag #
        ###############################
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
def seperate_delimiter(tokens):
    words = []
    delimiters = set(',.:;?!')

    for t in tokens:
        end = len(t) - 1

        if (t in delimiters) or (t[end] not in delimiters):
            words.append(t)
        else:
            words.append(t[:end])
            words.append(t[end])

    # return type: processed string
    return ' '.join(words)


##################################################
# function to further filter extracted json file #
##################################################
def filter_data(fname, path):
    f = open(path + fname)
    data = json.load(f)
    f.close()

    filtered = {'data': []}
    for d in data['data']:
        entities = filter_entities(d['text'], d['dict'])
        if len(entities) > 1:
            filtered['data'].append({'text': d['text'], 'dict': entities})

    print '\t {0} filtered\n'.format(fname)
    g = open(path + fname, 'w')
    json.dump(filtered, g, indent = 4)
    g.close()


####################################################
# function to filter entities based on parsed text #
####################################################
def filter_entities(text, dictionary):
    result = {}
    words = None
    try:
        words = set(text.split())
    except:
        return result

    for key, value in dictionary.iteritems():
        if key in words:
            result[key] = value

    # return type: filtered entity dictionary
    return result


#################################
# function to extract all guids #
#################################
def extract_guids(fnames, loadp, savep):
    entset = set([])
    guid2text = {}

    for i in range(0, len(fnames)):
        print 'On file {0}\n'.format(fnames[i])
        f = open(loadp + fnames[i], 'r')
        data = json.load(f)
        f.close()

        for d in data['data']:
            for key, value in d['dict'].iteritems():
                entset.add(value['freebase_id'])
                guid2text[value['freebase_id']] = value['anchor_text']

    entset = list(entset)
    entset.sort()
    guid = {'guid': entset}
    print '# guids: {0}'.format(len(entset))

    '''
    print '\t guid.json saved'
    g = open(savep + 'guid.json', 'w')
    json.dump(guid, g, indent = 4)
    g.close()
    '''

    print '\t guid2name.json saved'
    h = open(savep + 'guid2name.json', 'w')
    json.dump(guid2text, h, indent = 4)
    h.close()


###################################################
# function to compute basic statistics of dataset #
###################################################
def compute_stats(fnames, path):
    ################
    # global stats #
    ################
    num_docs = 0
    entset = set([])
    wordset = set([])
    ###############
    # local stats #
    ###############
    num_ents = 0
    num_words = 0

    for name in fnames:
        ##################
        # load data file #
        ##################
        print 'On file {0}'.format(name)
        f = open(path + name, 'r')
        data = json.load(f)
        f.close()

        ###########################
        # compute file statistics #
        ###########################
        for doc in data['data']:
            if _filtering and (len(doc['text'].split()) > _threshold):
                continue

            for key, value in doc['dict'].iteritems():
                entset.add(value['freebase_id'])
            for word in doc['text'].lower().split():
                wordset.add(word)

            num_docs += 1
            num_ents += len(doc['dict'])
            num_words += len(doc['text'].split())

    # return type: statistics dictionary
    return {'docs': num_docs, 'ent_voc': len(entset), 'word_voc': len(wordset),
            'ents': num_ents, 'words': num_words}


#################
# main function #
#################
if __name__ == '__main__':
    guidpath = '/scratch/data/freebase/'
    rawpath = '/scratch/data/wikilink/raw/'
    extpath = '/scratch/data/wikilink/ext/'

    ###################
    # option: extract #
    ###################
    if sys.argv[1] == '-extract':
        fnames = os.listdir(rawpath)
        fnames.sort()

        for name in fnames:
            print '****** processing {0} ****** \n'.format(name)
            extract_data(name, rawpath, extpath)

    ##################
    # option: filter #
    ##################
    if sys.argv[1] == '-filter':
        fnames = os.listdir(extpath)
        fnames.sort()

        for name in fnames:
            print '****** filtering {0} ****** \n'.format(name)
            filter_data(name, extpath)

    ################
    # option: guid #
    ################
    if sys.argv[1] == '-guid':
        fnames = os.listdir(extpath)
        fnames.sort()

        print '****** Extracting guids ******'
        extract_guids(fnames, extpath, guidpath)

    #################
    # option: count #
    #################
    if sys.argv[1] == '-count':
        fnames = os.listdir(extpath)
        fnames.sort()

        stats = compute_stats(fnames, extpath)

        print '# documents: {0}; # entities: {1}; vocabulary size: {2}'.format(stats['docs'], stats['ent_voc'], stats['word_voc'])
        print '\t - tokens / doc: {0}'.format(stats['words'] / float(stats['docs']))
        print '\t - entities / doc: {0}'.format(stats['ents'] / float(stats['docs']))
