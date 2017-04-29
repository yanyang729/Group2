from helper import *
from sklearn.base import BaseEstimator,TransformerMixin
import pandas as pd
import itertools
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.pipeline import Pipeline
from helper import normalize_corpus
import  gensim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import RandomizedSearchCV,cross_val_score,StratifiedShuffleSplit,train_test_split
from sklearn.metrics import recall_score,accuracy_score
import numpy as np
from tqdm import tqdm
from gensim import models,corpora
from sklearn.svm import SVC
import xgboost as xgb



class ParsedDataTransformer(BaseEstimator,TransformerMixin):
    """
    transform form parsed xml file into topn df
    df.content is a string sentence which is not cleaned
    df.tags is a list of lists only have top n tags
    """
    def __init__(self, topn_tags):
        self.topn_tags = topn_tags

    def fit(self,X,y=None):
        return self

    def transform(self,X,y=None):
        X.columns = ['title', 'body', 'tags']
        merged = [title + ' ' + body for title, body in zip(X.title, X.body)]
        df_merged = pd.DataFrame({'content': merged, 'tags': X.tags.copy()})
        data_tags = df_merged.tags.apply(lambda x: x.replace('<','').split('>')[:-1])
        new_targets = []
        for sample_tags in data_tags:
            sample_tags_wanted = []
            for tag in sample_tags:
                if tag in self.topn_tags:
                    sample_tags_wanted.append(tag)
            new_targets.append(sample_tags_wanted)
        bool_index = [True if t != [] else False for t in new_targets]
        rslt = pd.DataFrame({'content': df_merged.content, 'tags': new_targets}).loc[bool_index, :]
        return rslt.reset_index(drop=True)


class ContentCleaner(BaseEstimator,TransformerMixin):
    """
    using helper function to clean the content text
    """
    def __init__(self):
        pass

    def fit(self,X,y=None):
        return self

    def transform(self,X,y=None):
        # fully cleaned text
        noramalized_docs = normalize_corpus(X.content,tokenize=False)
        return pd.DataFrame({'content':noramalized_docs,'tags':X.tags})


class ExtractTfidfAveVec(BaseEstimator, TransformerMixin):
    """
    get the word to vector model, tfidf model, use them to get the sentence vector
    """
    def __init__(self,vec_len):
        self.vec_len = vec_len

    def fit(self, X, y=None):
        self.corpus = list(X.content.values.flatten())
        self.g_model = gensim.models.Word2Vec([s.split() for s in self.corpus], size=self.vec_len, window=10, min_count=2, sample=1e-3)

        return self

    def transform(self, X, y=None):
        return self.ave_weighted_tfidf(X.content.values.flatten())

    def tfidf_extractor(self, ngram_range=(1, 1)):

        tfidf_obj = TfidfVectorizer(min_df=3, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
                                    ngram_range=ngram_range, use_idf=1, smooth_idf=1, sublinear_tf=1,
                                    stop_words='english')

        tfidf_features = tfidf_obj.fit_transform(self.corpus)

        return tfidf_obj, tfidf_features

    def tfidf_mapper(self, tfidf_obj, tfidf_features):
        vocab = tfidf_obj.vocabulary_
        words = vocab.keys()
        word_tfidfs = [tfidf_features[0, vocab.get(word)] if vocab.get(word) else 0 for word in words]
        word_tfidf_map = {word: tfidf_val for word, tfidf_val in zip(words, word_tfidfs)}

        return word_tfidf_map

    def ave_weighted_tfidf(self, tokenized_list):
        self.tokenized_list = tokenized_list
        weighted_ave = []
        tfidf_obj, tfidf_features = self.tfidf_extractor()
        word_tfidf_map = self.tfidf_mapper(tfidf_obj, tfidf_features)

        for sentence in tqdm(self.tokenized_list):
            word_vecs = np.array(
                [self.g_model[word] * self.word_in_word_tfidf_map(word, word_tfidf_map) for word in sentence.split() if
                 word in self.g_model])

            weighted_ave.append(np.sum(word_vecs, axis=0) / len(word_vecs))

        return np.array(weighted_ave)

    def word_in_word_tfidf_map(self, word, word_tfidf_map):
        if word in word_tfidf_map.values():
            return word_tfidf_map[word]
        else:
            return 1


class LDATransformer(BaseEstimator, TransformerMixin):
    def __init__(self, num_topics, passes):
        self.num_topics = num_topics
        self.passes = passes

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        texts = X.content.apply(lambda x: x.split())
        dictionary = corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]
        print 'training LDA...'
        ldamodel = models.ldamodel.LdaModel(
            corpus, num_topics=self.num_topics, id2word=dictionary, passes=self.passes)
        dict_values = {i: [] for i in range(10)}
        for sample in ldamodel.get_document_topics(corpus):
            row = np.zeros(self.num_topics)
            for topic_id, value in sample:
                row[topic_id] = value
            for i, v in enumerate(row):
                dict_values[i].append(v)
        return pd.concat((pd.DataFrame(dict_values), X.iloc[:, 1]), axis=1)


def xgb_random_search(x_train,y_train,num_iter=10):
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


def svm_random_search(x_train, y_train, num_iter=10):
    param_distribs = {
        'estimator__kernel':[ 'linear', 'poly', 'rbf', 'sigmoid'],
        'estimator__C' : list(np.linspace(0.01,1,20)),
        "estimator__degree": [1, 2, 3, 4],
    }
    ovsr = OneVsRestClassifier(SVC(),n_jobs=4)
    rnd_search = RandomizedSearchCV(ovsr, param_distribs, n_iter=num_iter, cv=5)
    rnd_search.fit(x_train, y_train)

    cvres = rnd_search.cv_results_
    scores_list = []
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        scores_list.append((mean_score, params))
    highest_score, highest_paras = sorted(scores_list, key=lambda x: x[0], reverse=True)[0]
    print('best score: {}, best paras:{}'.format(highest_score, highest_paras))
    return highest_score,highest_paras


class TargetBinerizer(BaseEstimator,TransformerMixin):
    """
    transform the target into one-hot like vector
    """
    def __init__(self):
        pass

    def fit(self,X,y=None):
        return self

    def transform(self,X,y=None):
        mlb = MultiLabelBinarizer().fit(X.tags)
        y = mlb.transform(X.tags)
        return y


if __name__ == '__main__':

    df = pd.read_csv('./data/processed/stack_ds_4_9_2017 .csv', quotechar='|', sep=',', header=None)
    topn_tags = ['javascript', 'java', 'android', 'php', 'python', 'c#', 'html', 'jquery', 'ios', 'css']


    ppl_X = Pipeline([
        ('transformer',ParsedDataTransformer(topn_tags)),
        ('textCleaner',ContentCleaner()),
        ('doc2vecTFIDFtransformer',ExtractTfidfAveVec(vec_len=15))
    ])

    ppl_y = Pipeline([
        ('transformer',ParsedDataTransformer(topn_tags)),
        ('targetBinerizer',TargetBinerizer())
    ])


    X = ppl_X.fit_transform(df)
    y = ppl_y.fit_transform(df)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    _, best_para = svm_random_search(x_train,y_train)

    ovsr = OneVsRestClassifier(SVC(
        C=best_para['estimator__C'],
        kernel=best_para['estimator__kernel'],
        degree=best_para['estimator__degree']
    ), n_jobs=4).fit(x_train,y_train)

    preds = ovsr.predict(x_test)
    print 'recall score: {}'.format(recall_score(y_test, preds, average='weighted'))
    print 'accuracy score: {}'.format(accuracy_score(y_test, preds))
