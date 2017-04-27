# pandas for data manipulation
import pandas as pd
# nltk for nlp
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
# list of stopwords like articles, preposition
stop = set(stopwords.words('english'))
from string import punctuation
import re
# from nltk.stem.porter import PorterStemmer as Stemmer
from nltk.stem.snowball import SnowballStemmer as Stemmer


def stem_token(df):
    stemmer = Stemmer('english').stem
    tokens_list=[]
    for tokens in df:
        tokens_list.append(map(stemmer,tokens))
    return tokens_list


def tokenizer(text):
    try:
        tokens_ = [word_tokenize(sent) for sent in sent_tokenize(text)]

        tokens = []
        for token_by_sent in tokens_:
            tokens += token_by_sent

        tokens = list(filter(lambda t: t.lower() not in stop, tokens))
        tokens = list(filter(lambda t: t not in punctuation, tokens))

        tokens = list(filter(lambda t: t not in [u"'s", u"n't", u"...", u"''", u'``',u"'m"], tokens))
        filtered_tokens = []
        for token in tokens:
            if re.search('[a-zA-Z]', token):
                filtered_tokens.append(token)

        filtered_tokens = list(map(lambda token: token.lower(), filtered_tokens))
        filtered_tokens = stem_token(filtered_tokens)

        return filtered_tokens
    except Exception as e:
        print(e)