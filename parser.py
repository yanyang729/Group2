import sys
import os
import re
import csv
from dateutil import parser as dateparser
from operator import itemgetter
from xml.etree import cElementTree as etree
from collections import defaultdict
from tqdm import tqdm

READ_HOW_MANY = 100000
counter = defaultdict(int)


# parse body
def clean_body(s):
    code_match = re.compile('<pre>(.*?)</pre>', re.MULTILINE | re.DOTALL)
    link_match = re.compile('<a href="http://.*?".*?>(.*?)</a>', re.MULTILINE | re.DOTALL)
    img_match = re.compile('<img(.*?)/>', re.MULTILINE | re.DOTALL)
    tag_match = re.compile('<[^>]*>', re.MULTILINE | re.DOTALL)
    # remove code? link? img? tag?
    for match_str in code_match.findall(s):
        s = code_match.sub("", s)
    links = link_match.findall(s)
    s = re.sub(" +", " ", tag_match.sub('', s)).replace("\n", "")
    for link in links:
        if link.lower().startswith("http://"):
            s = s.replace(link, '')
    return s


# Assume all we need is title, body,tags
def parsexml(filename):
    it = map(itemgetter(1),
             iter(etree.iterparse(filename, events=('start',))))
    root = next(it)  # get posts element
    for elem in tqdm(it):
        if counter['all'] == READ_HOW_MANY :
            break
        counter['all'] += 1
        if elem.tag == 'row' and int(elem.get('PostTypeId')) == 1 :  # if row and is question
            title = elem.get('Title')
            body =  clean_body(elem.get('Body'))
            tags = elem.get('Tags')
            if title and body and tags:  # if not None
                counter['samples'] += 1
                one_question = [title,body,tags]
                yield one_question

filename = 'preprocesse_samples_from_first_{}.csv'.format(READ_HOW_MANY)

with open(filename, 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for values in parsexml('./Posts.xml'):
        writer.writerow(values)


print('Finish! how many questions we have?',counter['samples'] )

