from glob2 import glob
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator,TransformerMixin
from gensim import models,corpora
import numpy as np
import time
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score,RandomizedSearchCV
import xgboost as xgb
from sklearn.metrics import recall_score
import pickle
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

# TO DO:
# Build a ppl class which take into parsed xml data in to one binary target dataframe
def cleaner(sentence):
    to_be_removed = set(stopwords.words('english'))
    words = [word.lower() for word in word_tokenize(re.sub("[^a-zA-Z]"," ",sentence)) if word.lower() not in to_be_removed]
    nouns =  TextBlob(' '.join([w for w in words])).noun_phrases
    final_sentence = ' '.join([n for n in nouns])
    return final_sentence

class LDATransformer(BaseEstimator, TransformerMixin):
    def __init__(self, num_topics, passes):
        self.num_topics = num_topics
        self.passes = passes

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        tic = time.time()
        texts = X.content.apply(lambda x: x.split())
        dictionary = corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]
        print('training LDA...',end=' ')
        ldamodel = models.ldamodel.LdaModel(
            corpus, num_topics=self.num_topics, id2word=dictionary, passes=self.passes)
        dict_values = {i: [] for i in range(10)}
        for sample in ldamodel.get_document_topics(corpus):
            row = np.zeros(self.num_topics)
            for topic_id, value in sample:
                row[topic_id] = value
            for i, v in enumerate(row):
                dict_values[i].append(v)
        print('{} seconds used'.format(str(int(time.time() - tic))))
        return pd.concat((pd.DataFrame(dict_values), X.iloc[:, 1]), axis=1)

def my_train_test_split(df):
    X = df.iloc[:, :-1].values
    Y = df.iloc[:, -1].values
    skf = StratifiedShuffleSplit(n_splits=1, random_state=123)
    x_train, y_train, x_test, y_test = None,None,None,None,
    for train_i, test_i in skf.split(X, Y):
        x_train, y_train = X[train_i], Y[train_i]
        x_test, y_test = X[test_i], Y[test_i]
    return x_train,y_train,x_test,y_test


def random_search(x_train,y_train,num_iter=10):
    param_distribs = {
        'max_depth': [3, 4],
        'learning_rate': [0.01, 0.05, 0.1, 0.5],
        'n_estimators': list(range(100,1000,100)),
        'colsample_bytree': [0.5,0.7,0.9],
    }

    gbm = xgb.XGBClassifier()
    rnd_search = RandomizedSearchCV(gbm, param_distribs, n_iter=num_iter, cv=5, scoring='recall')
    rnd_search.fit(x_train, y_train)

    cvres = rnd_search.cv_results_
    scores_list = []
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        scores_list.append((mean_score, params))
    highest_score, highest_paras = sorted(scores_list, key=lambda x: x[0], reverse=True)[0]
    print('best score: {}, best paras:{}'.format(highest_score, highest_paras))
    return highest_score,highest_paras


if __name__ == '__main__':
    # to be tuned:
    # LDA model
    NUM_TOPICS = 10
    PASSES = 3
    # random search
    NUM_SEARCH = 20

    files = glob('./data/processed/10sets/*')
    for file in files:
        df = pd.read_csv(file)
        tag_name = file.split('/')[-1].split('.')[0]

        ppl = Pipeline([
            ('LDA', LDATransformer(num_topics=NUM_TOPICS, passes=PASSES)),
        ])
        df_ = ppl.fit_transform(df)

        x_train, y_train, x_test, y_test = my_train_test_split(df_)

        _, highest_paras =random_search(x_train,y_train,num_iter=NUM_SEARCH)

        # train a model and test on test data
        gbm = xgb.XGBClassifier(
            max_depth=highest_paras['max_depth'],
            n_estimators=highest_paras['n_estimators'],
            learning_rate=highest_paras['learning_rate'],
            colsample_bytree=highest_paras['colsample_bytree']
        )

        gbm_model = gbm.fit(x_train,y_train)

        pickle.dump(gbm_model,'./data/xgb_{}.pkl'.format(tag_name))
        preds = gbm_model.predict(x_test)
        print('final model test on test set. Score:{}'.format(recall_score(y_test,preds)))
        print('='*20)

        break

