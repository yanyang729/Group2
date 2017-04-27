# Stack Over Flow -  AutoTagging and Tag Optimizer
# BIA-660C Group - 2
# Members: Xianfei Gu, Yang Yang, Rush Kirubi, Jianjie Gao, Praveen Thinagarajan
# Functionality:
# Clean and Pre-Process StackOverFlow Data from XML to a cleaned CSV Format
# Find Top-N/Top-Frequent Tags and Consolidate Questions That Contain Those Tags into Separate CSV Files
#

import sys
import os
import re
import csv
from dateutil import parser as dateparser
from operator import itemgetter
from xml.etree import cElementTree as etree
from collections import defaultdict
from tqdm import tqdm
from collections import Counter
import pandas as pd
import re
import nltk
import string
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from collections import Counter
import pandas as pd
import time
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import warnings
warnings.filterwarnings('ignore')
stopword_list = stopwords.words('english')
wnl = WordNetLemmatizer()
from pattern.en import tag
from nltk.corpus import wordnet as wn


class StackOverFlowAutoTagger:
    counter = defaultdict(int)

    no_of_lines_to_be_processed = 3001
    stack_xml_location = './Posts.xml'
    pre_process_file_name_syntax = 'preprocesse_samples_from_first_{}.csv'
    number_of_tags = 10


    # Parsing and Cleaning done for body section of each StackOverFlow Question
    @staticmethod
    def clean_body_section(stack_data):
        # Regex Syntaxes to Detect Code,Links and Tags within Question
        code_match = re.compile('<pre>(.*?)</pre>', re.MULTILINE | re.DOTALL)
        link_match = re.compile('<a href="http://.*?".*?>(.*?)</a>', re.MULTILINE | re.DOTALL)
        tag_match = re.compile('<[^>]*>', re.MULTILINE | re.DOTALL)
        for match_str in code_match.findall(stack_data):
            stack_data = code_match.sub("", stack_data)
        links = link_match.findall(stack_data)
        # Replaces + Signs in the cleaner body section
        stack_data = re.sub(" +", " ", tag_match.sub('', stack_data)).replace("\n", "")
        # Removing the links within the body section
        for link in links:
            if link.lower().startswith("http://"):
                stack_data = stack_data.replace(link, '')
        return stack_data

    # Parse XML file to return Title, Body and Tag of the Stack Overflow Question
    @staticmethod
    def parse_xml(stack_data_xml_file_name):
        it = map(itemgetter(1), iter(etree.iterparse(stack_data_xml_file_name, events=('start',))))
        for elem in tqdm(it):
            if StackOverFlowAutoTagger.counter['all'] == StackOverFlowAutoTagger.no_of_lines_to_be_processed:
                break
            StackOverFlowAutoTagger.counter['all'] += 1
            # Elements in XML can have a Tag of either 'Post' or 'Row'
            # Checking if the element is 'Row' and element type is a 'Question'
            if elem.tag == 'row' and int(elem.get('PostTypeId')) == 1:
                title = elem.get('Title')
                body = StackOverFlowAutoTagger.clean_body_section(elem.get('Body'))
                tags = elem.get('Tags')
                # Checking if Title,Body and Tags are available for the Question
                if title and body and tags:
                    StackOverFlowAutoTagger.counter['samples'] += 1
                    one_question = [title, body, tags]
                    yield one_question

    @staticmethod
    def get_top_n_tags(path, topn):
        df = pd.read_csv(path, quotechar='|', sep=',', header=None)
        df.columns = ['title', 'body', 'tags']
        merged = [title + ' ' + body for title, body in zip(df.title, df.body)]
        df_merged = pd.DataFrame({'content': merged, 'tags': df.tags})
        df_merged.tags = df_merged.tags.apply(lambda x: x.replace('<', '').split('>')[:-1])
        df_transfromed = pd.DataFrame(df_merged.tags.tolist(), index=df_merged.content).stack().reset_index()[
            ['content', 0]]
        df_transfromed.columns = ['content', 'tags']
        top_tags = Counter(df_transfromed.tags).most_common()[:topn]
        top_n_tags = [fktag for fktag, num in top_tags]
        return top_n_tags

    @staticmethod
    def get_topn_tags_transform(path, topn, tag):
        """
        read cleaned data and transform them into one tag per row
        1. get top n tags
        2. expand row
        3. to boolean
        4. aggregate by content
        """
        df = pd.read_csv(path, quotechar='|', sep=',', header=None)
        df.columns = ['title', 'body', 'tags']
        merged = [title + ' ' + body for title, body in zip(df.title, df.body)]
        df_merged = pd.DataFrame({'content': merged, 'tags': df.tags})
        df_merged.tags = df_merged.tags.apply(lambda x: x.replace('<', '').split('>')[:-1])
        df_transfromed = pd.DataFrame(df_merged.tags.tolist(), index=df_merged.content).stack().reset_index()[
            ['content', 0]]
        df_transfromed.columns = ['content', 'tags']
        top_tags = Counter(df_transfromed.tags).most_common()[:topn]
        top_n_tags = [fktag for fktag, num in top_tags]
        df_filtered = df_transfromed[df_transfromed.tags.apply(lambda x: x in set(top_n_tags))]
        df_filtered.tags = [int(bool) for bool in df_filtered.tags == tag]
        df_filtered.columns = ['content', 'is_{}'.format(tag)]
        rslt = df_filtered.groupby('content')['is_{}'.format(tag)].agg(['sum']).reset_index()
        rslt.columns = ['content', 'is_{}'.format(tag)]
        return rslt, top_n_tags

    @staticmethod
    def tokenize_text(text):
        tokens = nltk.word_tokenize(text)
        tokens = [token.strip() for token in tokens]
        return tokens

    @staticmethod
    def expand_contractions(text, contraction_mapping):
        contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                          flags=re.IGNORECASE | re.DOTALL)

        def expand_match(contraction):
            match = contraction.group(0)
            first_char = match[0]
            expanded_contraction = contraction_mapping.get(match) \
                if contraction_mapping.get(match) \
                else contraction_mapping.get(match.lower())
            expanded_contraction = first_char + expanded_contraction[1:]
            return expanded_contraction

        expanded_text = contractions_pattern.sub(expand_match, text)
        expanded_text = re.sub("'", "", expanded_text)
        return expanded_text



    # Annotate text tokens with POS tags
    @staticmethod
    def pos_tag_text(text):
        def penn_to_wn_tags(pos_tag):
            if pos_tag.startswith('J'):
                return wn.ADJ
            elif pos_tag.startswith('V'):
                return wn.VERB
            elif pos_tag.startswith('N'):
                return wn.NOUN
            elif pos_tag.startswith('R'):
                return wn.ADV
            else:
                return None

        tagged_text = tag(text)
        tagged_lower_text = [(word.lower(), penn_to_wn_tags(pos_tag))
                             for word, pos_tag in
                             tagged_text]
        return tagged_lower_text

    # lemmatize text based on POS tags
    @staticmethod
    def lemmatize_text(text):
        pos_tagged_text = StackOverFlowAutoTagger.pos_tag_text(text)
        lemmatized_tokens = [wnl.lemmatize(word, pos_tag) if pos_tag
                             else word
                             for word, pos_tag in pos_tagged_text]
        lemmatized_text = ' '.join(lemmatized_tokens)
        return lemmatized_text

    @staticmethod
    def remove_special_characters(text):
        tokens = StackOverFlowAutoTagger.tokenize_text(text)
        pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
        filtered_tokens = filter(None, [pattern.sub('', token) for token in tokens])
        filtered_text = ' '.join(filtered_tokens)
        return filtered_text

    @staticmethod
    def remove_stopwords(text):
        tokens = StackOverFlowAutoTagger.tokenize_text(text)
        filtered_tokens = [token for token in tokens if token not in stopword_list]
        filtered_text = ' '.join(filtered_tokens)
        return filtered_text

    @staticmethod
    def normalize_corpus(corpus, tokenize=False):
        normalized_corpus = []
        for text in corpus:
            text = StackOverFlowAutoTagger.expand_contractions(text, CONTRACTION_MAP)
            text = StackOverFlowAutoTagger.lemmatize_text(text)
            text = StackOverFlowAutoTagger.remove_special_characters(text)
            text = StackOverFlowAutoTagger.remove_stopwords(text)
            normalized_corpus.append(text)
            if tokenize:
                text = StackOverFlowAutoTagger.tokenize_text(text)
                normalized_corpus.append(text)

        return normalized_corpus

CONTRACTION_MAP = {
        "ain't": "is not",
        "aren't": "are not",
        "can't": "cannot",
        "can't've": "cannot have",
        "'cause": "because",
        "could've": "could have",
        "couldn't": "could not",
        "couldn't've": "could not have",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hadn't've": "had not have",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'd've": "he would have",
        "he'll": "he will",
        "he'll've": "he he will have",
        "he's": "he is",
        "how'd": "how did",
        "how'd'y": "how do you",
        "how'll": "how will",
        "how's": "how is",
        "I'd": "I would",
        "I'd've": "I would have",
        "I'll": "I will",
        "I'll've": "I will have",
        "I'm": "I am",
        "I've": "I have",
        "i'd": "i would",
        "i'd've": "i would have",
        "i'll": "i will",
        "i'll've": "i will have",
        "i'm": "i am",
        "i've": "i have",
        "isn't": "is not",
        "it'd": "it would",
        "it'd've": "it would have",
        "it'll": "it will",
        "it'll've": "it will have",
        "it's": "it is",
        "let's": "let us",
        "ma'am": "madam",
        "mayn't": "may not",
        "might've": "might have",
        "mightn't": "might not",
        "mightn't've": "might not have",
        "must've": "must have",
        "mustn't": "must not",
        "mustn't've": "must not have",
        "needn't": "need not",
        "needn't've": "need not have",
        "o'clock": "of the clock",
        "oughtn't": "ought not",
        "oughtn't've": "ought not have",
        "shan't": "shall not",
        "sha'n't": "shall not",
        "shan't've": "shall not have",
        "she'd": "she would",
        "she'd've": "she would have",
        "she'll": "she will",
        "she'll've": "she will have",
        "she's": "she is",
        "should've": "should have",
        "shouldn't": "should not",
        "shouldn't've": "should not have",
        "so've": "so have",
        "so's": "so as",
        "that'd": "that would",
        "that'd've": "that would have",
        "that's": "that is",
        "there'd": "there would",
        "there'd've": "there would have",
        "there's": "there is",
        "they'd": "they would",
        "they'd've": "they would have",
        "they'll": "they will",
        "they'll've": "they will have",
        "they're": "they are",
        "they've": "they have",
        "to've": "to have",
        "wasn't": "was not",
        "we'd": "we would",
        "we'd've": "we would have",
        "we'll": "we will",
        "we'll've": "we will have",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'll": "what will",
        "what'll've": "what will have",
        "what're": "what are",
        "what's": "what is",
        "what've": "what have",
        "when's": "when is",
        "when've": "when have",
        "where'd": "where did",
        "where's": "where is",
        "where've": "where have",
        "who'll": "who will",
        "who'll've": "who will have",
        "who's": "who is",
        "who've": "who have",
        "why's": "why is",
        "why've": "why have",
        "will've": "will have",
        "won't": "will not",
        "won't've": "will not have",
        "would've": "would have",
        "wouldn't": "would not",
        "wouldn't've": "would not have",
        "y'all": "you all",
        "y'all'd": "you all would",
        "y'all'd've": "you all would have",
        "y'all're": "you all are",
        "y'all've": "you all have",
        "you'd": "you would",
        "you'd've": "you would have",
        "you'll": "you will",
        "you'll've": "you will have",
        "you're": "you are",
        "you've": "you have"
    }


# ****************Start of pre-processed Data CSV creation ********************

# Create an Object for StackOverFlowAutoTagger
# Set number_of_lines to be processed, xml file location, pre-processing file syntax
# stack_object = StackOverFlowAutoTagger(number_of_lines=500,stack_xml_file='./Posts.xml',pre_filename='preprocesse_samples_from_first_{}.csv')

# Create pre-processed data filename
csv_filename = StackOverFlowAutoTagger.pre_process_file_name_syntax.format(StackOverFlowAutoTagger.no_of_lines_to_be_processed)

# Create Pre-Processing CSV file consisting of specified number of StackOverFlow Lines
with open(csv_filename, 'w') as csv_file:
    writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for values in StackOverFlowAutoTagger.parse_xml(StackOverFlowAutoTagger.stack_xml_location):
        try:
            writer.writerow(values)
        except:
            continue
print('Finish! how many questions we have?',StackOverFlowAutoTagger.counter['samples'])

# *****************End of pre-processed Data CSV creation   ***********************************


# ****************Start of finding Top-N Tags and CSV creation for each Tag ********************
time_taken = time.time()
pre_processed_data_path = './' + csv_filename
number_of_tags_considered = StackOverFlowAutoTagger.number_of_tags
print('start transform data')
tags_considered =  StackOverFlowAutoTagger.get_top_n_tags(pre_processed_data_path, number_of_tags_considered)
for i in range(10):
    considered_tag_name = tags_considered[i]
    df, _ = StackOverFlowAutoTagger.get_topn_tags_transform(pre_processed_data_path, number_of_tags_considered, considered_tag_name)
    corpus = [text for text in df.content]
    corpus = StackOverFlowAutoTagger.normalize_corpus(corpus)
    df.content = corpus
    df.to_csv('./input/is_{}.csv'.format(considered_tag_name), index=False)
    toc = time.time()
    print('{}s used'.format(str(int(toc - time_taken))))

# ****************End of finding Top-N Tags and CSV creation for each Tag ********************