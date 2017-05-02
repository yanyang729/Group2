from helper import *
from sklearn.base import BaseEstimator,TransformerMixin
import pandas as pd
import itertools
from collections import Counter, OrderedDict
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.pipeline import Pipeline
from helper import normalize_corpus
import gensim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import RandomizedSearchCV,cross_val_score,StratifiedShuffleSplit,train_test_split
from sklearn.metrics import recall_score,accuracy_score,make_scorer
import numpy as np
from tqdm import tqdm
from gensim import models,corpora
from sklearn.svm import SVC
import xgboost as xgb
import math



class ParsedDataTransformer(BaseEstimator,TransformerMixin):
    """
    transform form parsed xml file into topn df
    df.content is a string sentence which is not cleaned
    df.tags is a list of SINGLE TAG only have top n tags
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


class DropRowWithMultipleTags(BaseEstimator,TransformerMixin):
    """
    to reduce overlap, we drop those question with multiple top N tags
    """
    def __init__(self):
        pass

    def fit(self,X,y=None):
        return self

    def transform(self,X,y=None):
        mask = []
        for row in X.tags:
            if len(row) == 1:
                mask.append(True)
            else:
                mask.append(False)
        return X[mask]



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


class BOWVector(BaseEstimator,TransformerMixin):
    """
    to bag of words vector
    """
    def __init__(self,topn_tags):
        self.topn_tags = topn_tags

    def fit(self,X,y=None):
        return self

    def transform(self,X,y=None):
        content = ' '.join(list(X.content))
        uniq_words = set(content.split())
        termsTagRatioList = self.TermOccuranceRatioFinder(X,self.topn_tags)
        word_to_int = { word:i for i,word in enumerate(termsTagRatioList)}
        # int_to_word = {i:word for i,word in enumerate(termsTagRatioList)}
        content_vecs = []
        for sentence in X.content:
            vecs = np.zeros(len(termsTagRatioList))
            for word in sentence.split():
                if word in termsTagRatioList:
                    vecs[word_to_int[word]] += 1
                    if word in topn_tags:
                        vecs[word_to_int[word]] *= 77
            content_vecs.append(vecs)
        return np.array(content_vecs)

    @classmethod
    def TermOccuranceRatioFinder(cls,X,topn_tags):
        positive_counts = Counter()
        negative_counts = Counter()
        total_counts = Counter()

        tf_unique_words = []
        for tag_id in range(len(topn_tags)):
            for i in range(len(X.content)):
                try:
                    if (X.tags[i][0] == topn_tags[tag_id]):
                        for word in X.content[i].split():
                            positive_counts[word] += 1
                            total_counts[word] += 1
                    else:
                        for word in X.content[i].split():
                            negative_counts[word] += 1
                            total_counts[word] += 1
                except:
                    pass

            print positive_counts.most_common()

            pos_neg_ratios = Counter()

            # Calculate the ratios of positive and negative uses of the most common words
            # Consider words to be "common" if they've been used at least 100 times
            for term, cnt in list(total_counts.most_common()):
                if (cnt > 20):
                    pos_neg_ratio = positive_counts[term] / float(negative_counts[term] + 1)
                    pos_neg_ratios[term] = pos_neg_ratio

            # Convert ratios to logs
            for word, ratio in pos_neg_ratios.most_common():
                if (ratio > 0):
                    pos_neg_ratios[word] = np.log(ratio)
                else:
                    pos_neg_ratios[word] = -np.log((1 / (ratio + 0.01)))

            print pos_neg_ratios.most_common()

            min_doc_frequency = -4
            counting = 0
            for term_index,term_ratio_tuple in enumerate(pos_neg_ratios.most_common()):
                term = term_ratio_tuple[0]
                ratio = term_ratio_tuple[1]
                if ratio > min_doc_frequency:
                # if counting < 500:
                    if term not in tf_unique_words:
                        tf_unique_words.append(term)
                counting += 1

        return tf_unique_words

            # Please Ignore the comments - To be deleted while review
            # final_term_list = []
            # header_list = ['Unique_Words'] + topntags
            # final_term_list.append(header_list)
            # tag_ratio_list = OrderedDict()
            # for i, tag in enumerate(topn_tags):
            #     top_words_list = []
            #     word_ratio_list = OrderedDict()
            #     for index, word in enumerate(uniqueWords):
            #         co_with_tag = 0
            #         co_without_tag = 0
            #         counting=0
            #         for value in X.content:
            #             try:
            #                 if tag == X.tags[counting][0]:
            #                     if word in value[counting].split():
            #                         co_with_tag += 1
            #                     else:
            #                         co_without_tag += 1
            #             except:
            #                 pass
            #             counting += 1
            #         div_value = co_with_tag/(co_without_tag+0.00001)
            #         if div_value > 0:
            #             to_ratio = float(math.log(div_value))
            #         else:
            #             to_ratio = 0
            #         word_ratio_list[word] = to_ratio
            #     top_words_list = sorted(word_ratio_list)[0:n_words]
            #     tag_ratio_list[tag]=top_words_list
            # return tag_ratio_list


class ExtractTfidfAveVec(BaseEstimator, TransformerMixin):
    """
    get the word to vector model, tfidf model, use them to get the sentence vector
    """
    def __init__(self,vec_len,topn_tags):
        self.vec_len = vec_len
        self.topn_tags = topn_tags

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

        #add more main language weight
        for tag in self.topn_tags:
            word_tfidf_map[tag] = 100*word_tfidf_map.get(tag,np.zeros(self.vec_len))
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


def xgb_random_search(x_train,y_train,num_iter=5):
    param_distribs = {
        'estimator__max_depth': [3, 4],
        'estimator__learning_rate': [0.01, 0.05, 0.1, 0.5],
        'estimator__n_estimators': list(range(100,1000,100)),
        'estimator__colsample_bytree': [0.5,0.7,0.9],
    }

    gbm = OneVsRestClassifier(xgb.XGBClassifier(),n_jobs=-1)
    def score_func(y, y_pred, **kwargs):
        return recall_score(y,y_pred,average='macro')
    rnd_search = RandomizedSearchCV(gbm, param_distribs, n_iter=num_iter, cv=5,scoring=make_scorer(score_func))
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
        "estimator__class_weight":['balanced'],
        "estimator__probability":[True]
    }

    ovsr = OneVsRestClassifier(SVC(),n_jobs=-1)

    def score_func(y, y_pred, **kwargs):
        return recall_score(y,y_pred,average='macro')

    rnd_search = RandomizedSearchCV(ovsr, param_distribs, n_iter=num_iter, cv=5,scoring=make_scorer(score_func))
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
        return y,mlb

def bool_to_int(array):
    rslt = []
    for b in array:
        rslt.append(int(b))
    return rslt

if __name__ == '__main__':

    df = pd.read_csv('./data/processed/stack_ds_4_9_2017 .csv', quotechar='|', sep=',', header=None )
    topn_tags = ['javascript', 'java', 'android', 'php', 'python', 'c#', 'html', 'jquery', 'ios', 'css']


    ppl_X = Pipeline([
        ('transformer',ParsedDataTransformer(topn_tags)),
        ('droprowwithgtonetag',DropRowWithMultipleTags()),
        ('textCleaner',ContentCleaner()),
        # ('doc2vecTFIDFtransformer',ExtractTfidfAveVec(vec_len=15,topn_tags=topn_tags)),
        ('BOWVector',BOWVector(topn_tags=topn_tags))
    ])

    ppl_y = Pipeline([
        ('transformer',ParsedDataTransformer(topn_tags)),
        ('droprowwithgtonetag', DropRowWithMultipleTags()),
        ('targetBinerizer',TargetBinerizer())
    ])


    X = ppl_X.fit_transform(df)
    y,mlb = ppl_y.fit_transform(df)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)

    # SVM
    # _, best_para = svm_random_search(x_train,y_train)
    #
    # ovsr = OneVsRestClassifier(SVC(
    #     C=best_para['estimator__C'],
    #     kernel=best_para['estimator__kernel'],
    #     degree=best_para['estimator__degree'],
    #     class_weight=best_para['estimator__class_weight'],
    #     probability=best_para["estimator__probability"]
    # ), n_jobs=-1).fit(x_train,y_train)
    #
    # preds = ovsr.predict_proba(x_test)
    # preds = np.array([ bool_to_int(y == np.max(y)) for y in preds])
    #
    # print mlb.inverse_transform(preds)
    # print 'recall score: {}'.format(recall_score(y_test, preds, average='macro'))
    # print 'accuracy score: {}'.format(accuracy_score(y_test, preds))

    # XGB
    # _, best_para = xgb_random_search(x_train, y_train,num_iter=3)

    ovsr = OneVsRestClassifier(xgb.XGBClassifier(silent=1),n_jobs=-1).fit(x_train,y_train)
    preds = ovsr.predict_proba(x_test)
    preds = np.array([bool_to_int(y == np.max(y)) for y in preds])

    print mlb.inverse_transform(preds)
    print 'recall score: {}'.format(recall_score(y_test, preds, average='macro'))
    print 'accuracy score: {}'.format(accuracy_score(y_test, preds))


# ===================result=======================
"""
recall score: 0.642362746282
accuracy score: 0.750636132316
"""

