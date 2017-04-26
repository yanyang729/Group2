import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
# from string import punctuation
from collections import Counter
import re
import numpy as np
# import itertools
# from sklearn.feature_extraction.text import TfidfVectorizer
# from nltk.stem.porter import PorterStemmer
# from nltk.stem.snowball import SnowballStemmer
from gensim import corpora, models
from textblob import TextBlob
# import warnings
# warnings.filterwarnings('ignore')
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold,StratifiedShuffleSplit
# import xgboost as xgb
# from sklearn.metrics import accuracy_score,recall_score
# from sklearn.linear_model import LogisticRegression


class StackOverFlowTagger(object):
    DATA_PATH = 'preprocesse_samples_from_first_100000.csv'
    FILTERED_TAG = 'java'
    NUM_TAGS_CONSIDERED = 10

    @staticmethod
    def get_topn_tags_transform(path,topn,tag):
        """
        read cleaned data and transform them into one tag per row
        1. get top n tags
        2. expand row
        3. to boolean
        4. aggregate by content
        """
        df = pd.read_csv(path,quotechar='|',sep=',',header=None)
        df.columns = ['title','body','tags']
        merged = [ title + ' ' + body for title, body in zip(df.title,df.body)]
        df_merged = pd.DataFrame({'content':merged,'tags':df.tags})
        df_merged.tags = df_merged.tags.apply(lambda x: x.replace('<','').split('>')[:-1])
        df_transfromed = pd.DataFrame(df_merged.tags.tolist(),index=df_merged.content).stack().reset_index()[['content',0]]
        df_transfromed.columns = ['content','tags']
        top_tags = Counter(df_transfromed.tags).most_common()[:topn]
        top_n_tags = [tag for tag, num in top_tags]
        df_filtered = df_transfromed[df_transfromed.tags.apply(lambda x: x in set(top_n_tags))]
        df_filtered.tags = [int(bool) for bool in df_filtered.tags == tag]
        df_filtered.columns = ['content','is_{}'.format(tag)]
        rslt = df_filtered.groupby('content')['is_{}'.format(tag)].agg(['sum']).reset_index()
        rslt.columns = ['content','is_{}'.format(tag)]
        return rslt, top_n_tags

    @staticmethod
    def cleaner(sentence):
        to_be_removed = set(stopwords.words('english'))
        words = [word.lower() for word in word_tokenize(re.sub("[^a-zA-Z]"," ",sentence)) if word.lower() not in to_be_removed]
        nouns =  TextBlob(' '.join([w for w in words])).noun_phrases
        final_sentence = ' '.join([n for n in nouns])
        return final_sentence

    @staticmethod
    def simpleLDA(df, num_topics=10, passes=3):
        texts = df.content.apply(lambda x: StackOverFlowTagger.cleaner(x).split(' '))
        dictionary = corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]
        ldamodel = models.ldamodel.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=passes)
        dict_values = {i: [] for i in range(10)}
        for sample in tqdm(ldamodel.get_document_topics(corpus)):
            row = np.zeros(num_topics)
            for topic_id, value in sample:
                row[topic_id] = value
            for i, v in enumerate(row):
                dict_values[i].append(v)
        return pd.concat((pd.DataFrame(dict_values), df.iloc[:, 1]), axis=1)

    @staticmethod
    def preprocessedDataForXGB(test_data_indicator,data_path,tag_name,number_of_tags):
        df, top_n_tags = StackOverFlowTagger.get_topn_tags_transform(data_path,number_of_tags,tag_name)
        # print df
        X = df.iloc[:,:-1].values
        Y = df.iloc[:,-1].values
        skf = StratifiedShuffleSplit(n_splits=1, random_state=123)
        for train_i,test_i in skf.split(X,Y):
            x_train,y_train = X[train_i], Y[train_i]
            x_test,y_test = X[test_i],Y[test_i]
        if test_data_indicator:
            return x_test,y_test
        else:
            return x_train, y_train

# gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05).fit(x_train, y_train)
# pred = gbm.predict(x_test)

# Get Training X and Y data
X_Train_Data,Y_Train_Data = StackOverFlowTagger.preprocessedDataForXGB(False,StackOverFlowTagger.DATA_PATH,StackOverFlowTagger.FILTERED_TAG, StackOverFlowTagger.NUM_TAGS_CONSIDERED )

# Get Test X and Y data
X_Test_Data,Y_Test_Data =  StackOverFlowTagger.preprocessedDataForXGB(True,StackOverFlowTagger.DATA_PATH,StackOverFlowTagger.FILTERED_TAG, StackOverFlowTagger.NUM_TAGS_CONSIDERED )

