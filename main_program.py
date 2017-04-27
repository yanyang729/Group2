import pandas as pd
import numpy as np
import sys
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import gensim
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import normalization


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
    df_merged = pd.DataFrame({'content':merged,'tags':df.tags.copy()})
    df_merged.tags = df_merged.tags.apply(lambda x: x.replace('<','').split('>')[:-1])
    df_transformed = pd.DataFrame(df_merged.tags.tolist(),index=df_merged.content).stack().reset_index()[['content',0]]
    df_transformed.columns = ['content','tags']
    top_tags = Counter(df_transformed.tags).most_common()[:topn]
    top_n_tags = [tag for tag, num in top_tags]
    df_filtered = df_transformed[df_transformed.tags.apply(lambda x: x in set(top_n_tags))]
    df_filtered.tags = [int(bool) for bool in df_filtered.tags == tag]
    df_filtered.columns = ['content','is_{}'.format(tag)]
    rslt = df_filtered.groupby('content')['is_{}'.format(tag)].agg(['sum']).reset_index()
    rslt.columns = ['content','is_{}'.format(tag)]
    return rslt, top_n_tags




class ExtractAverageWordVectors(BaseEstimator,TransformerMixin):
	def __init__(self,g_model):
		self.g_model = g_model

	def fit(self, X, y=None):
		return self

	def transform(self, X, y=None):
		if self.g_model:
			return self.averaged_word_vectors(X) 

	def averaged_word_vectors(self, tokenized_list):
		weighted_ave = []

    
		for sentence in tqdm(tokenized_list):
			word_vecs =  np.array([self.g_model[word] for word in sentence if word in self.g_model])
        
			weighted_ave.append(np.sum(word_vecs, axis = 0)/len(word_vecs))
    
		return np.array(weighted_ave)



class ExtractTfidfAveVec(BaseEstimator,TransformerMixin):
    def __init__(self,corpus, g_model):
        self.corpus = corpus
        self.g_model = g_model

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return self.ave_weighted_tfidf(X) 

    def tfidf_extractor(self, ngram_range=(1,1)):

        tfidf_obj = TfidfVectorizer(min_df=3, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}', ngram_range=ngram_range, use_idf = 1, smooth_idf = 1, sublinear_tf = 1, stop_words='english')

        tfidf_features = tfidf_obj.fit_transform(self.corpus)
        
        return tfidf_obj, tfidf_features

    def tfidf_mapper(self, tfidf_obj, tfidf_features):
        vocab = tfidf_obj.vocabulary_
        words = vocab.keys()
        word_tfidfs = [tfidf_features[0, vocab.get(word)] if vocab.get(word) else 0 for word in words]
        word_tfidf_map = {word:tfidf_val for word, tfidf_val in zip(words, word_tfidfs)}
        
        return word_tfidf_map

    def ave_weighted_tfidf(self, tokenized_list):
        self.tokenized_list = tokenized_list
        weighted_ave = []
        tfidf_obj, tfidf_features = self.tfidf_extractor()
        word_tfidf_map = self.tfidf_mapper(tfidf_obj, tfidf_features ) 

        for sentence in tqdm(self.tokenized_list):
            word_vecs =  np.array([self.g_model[word] * self.word_in_word_tfidf_map(word, word_tfidf_map)  for word in sentence if word in self.g_model])

            weighted_ave.append(np.sum(word_vecs, axis = 0)/len(word_vecs))

        return np.array(weighted_ave)
    
    def word_in_word_tfidf_map(self, word, word_tfidf_map):
        if word in word_tfidf_map.values():
            return word_tfidf_map[word]
        else: 
            return 1

class Doc2Vec(BaseEstimator,TransformerMixin):
	def __init__():
		pass
	def fit(self, X, y=None):
		return self

	def transform(self, X, y=None):
		pass



def main():
	print("="*32)
	print("STACKOVERFLOW QUESTION TAGGER")
	print("="*32)

	# question = input("Please enter the stackoverflow question to be tagged:\n ")

	# print(question)
	df, _ = get_topn_tags_transform("data/processed/stack_ds_4_9_2017 .csv",10,'java')

	
	X_train, X_test, y_train, y_test = train_test_split(df.ix[:,:-1], df.ix[:,-1], test_size=0.3, random_state=123)

	t = list(X_train[:].values.flatten())
	
	tokenized_text = normalization.normalize_corpus(t, True)
	
	model_word_two_vec = gensim.models.Word2Vec(tokenized_text, size=10, window=10, min_count=2, sample=1e-3)

	# eawv = ExtractAverageWordVectors(model_word_two_vec)
	wv_tfidf = ExtractTfidfAveVec(corpus=t, g_model=model_word_two_vec)

	X = wv_tfidf.transform(tokenized_text)
	y_train

	

	
	print(len(X))
	print(X[:4])
	


if __name__=="__main__":
	main()


