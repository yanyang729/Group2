import pandas as pd
import itertools
from collections import Counter
# nltk for nlp
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
# list of stopwords like articles, preposition
stop = set(stopwords.words('english'))
from string import punctuation
from collections import Counter
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from tokenizer import tokenizer
from sklearn.model_selection import train_test_split


class Preprocess:
    def __init__(self,content=None,tag=None,filepath=''):
        self.path=filepath
        if filepath!='':
            self.df = Preprocess.read_csv(filepath)
            self.df['content']=self.df.title+u' '+self.df.body
        elif content!=None and tag!=None:
            self.df = pd.DataFrame([content, tag], columns=['content', 'tags'])
        else:
            raise ValueError("No suitable argument")

    @staticmethod
    def read_csv(path):
        df = pd.read_csv(path, sep=',', quotechar='|', header=None, encoding='utf-8')
        df.columns = ['title', 'body', 'tags']
        return df

    @staticmethod
    def tag_transforms(df_tags):
        all_tags = df_tags.apply(lambda x: x.replace('<', '').split('>')[:-1])
        all_tags = [x for x in list(itertools.chain(*all_tags)) if x]
        ct = Counter(all_tags)
        vocab = ct.most_common()
        pd_tag = pd.DataFrame(vocab[:10])
        tag_used = [x[0] for x in pd_tag]
        tag_result = []
        for tags in all_tags:
            tag_result.append([x for x in tags if x in tag_used])
        return tag_result

    @staticmethod
    def process_X(df, max_features=100):
        vectorizer = TfidfVectorizer(tokenizer=tokenizer, max_features=max_features)
        X_train = vectorizer.fit_transform(df)
        return X_train

    @staticmethod
    def process_Y(df):
        tag_result = Preprocess.tag_transforms(df)
        mlb = MultiLabelBinarizer()
        Y = mlb.fit_transform(tag_result)

    """
    :return X_train, X_test, Y_train, Y_test
    """
    def preprocess(self):

        X_train, X_test, Y_train, Y_test=train_test_split(self.df.content, self.df.tags, test_size=0.30, random_state=123)
        X_train=Preprocess.process_X(X_train)
        X_test = Preprocess.process_X(X_test)
        Y_train = Preprocess.process_Y(Y_train)
        Y_test = Preprocess.process_Y(Y_test)

        return  X_train, X_test, Y_train, Y_test