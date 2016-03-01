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
    f = open(savepath + '{0}.json'.format(fname[:3]), 'w')
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
    filtered = []
    f.close()

    for d in data['data']:
        entities = filter_entities(d['text'], d['dict'])
        if len(entities) > 1:
            filtered.append({'text': d['text'], 'dict': entities})

    g = open(path + fname, 'w')
    data['data'] = filtered
    json.dump(data, g, indent = 4)
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


###################################################
# function to compute basic statistics of dataset #
###################################################
def compute_stats(fnames, path):
    ################
    # global stats #
    ################
    docs = 0
    entset = set([])
    wordset = set([])
    ###############
    # local stats #
    ###############
    ents = 0
    words = 0

    for i in range(0, len(fnames)):
        ##################
        # load data file #
        ##################
        f = open(path + fnames[i], 'r')
        data = json.load(f)
        f.close()

        ###########################
        # compute file statistics #
        ###########################
        docs += len(data['data'])
        for d in data['data']:
            for key, value in d['dict'].iteritems():
                entset.add(value['freebase_id'])
            for w in d['text'].split():
                wordset.add(w)
            ents += len(d['dict'])
            words += len(d['text'].split())

    # return type: statistics dictionary
    return {'docs': docs, 'ent_voc': len(entset), 'word_voc': len(wordset),
            'ents': ents, 'words': words}


#################
# main function #
#################
if __name__ == '__main__':
    loadpath = '/scratch/data/wikilink/raw/'
    savepath = '/scratch/data/wikilink/ext/'

    ###################
    # option: extract #
    ###################
    if sys.argv[1] == '-extract':
        fnames = os.listdir(loadpath)
        fnames.sort()

        for i in range(int(sys.argv[2]), int(sys.argv[3])):
            print '****** processing {0} ****** \n'.format(fnames[i])
            extract_data(fnames[i], loadpath, savepath)

    ##################
    # option: filter #
    ##################
    if sys.argv[2] == '-filter':
        fnames = os.listdir(savepath)
        fnames.sort()

        for i in range(0, len(fnames)):
            print '****** filtering {0} ****** \n'.format(fnames[i])
            filter_data(fnames[i], savepath)

    #################
    # option: count #
    #################
    if sys.argv[3] == '-count':
        fnames = os.listdir(savepath)
        fnames.sort()

        stats = compute_stats(fnames, savepath)
        print '# documents: {0}; # entities: {1}; vocabulary size: {2}'.format(stats['docs'], stats['ent_voc'], stats['word_voc'])
        print '\t - tokens / doc: {0}'.format(stats['words'] / float(stats['docs']))
        print '\t - entities / doc: {0}'.format(stats['ents'] / float(stats['docs']))
