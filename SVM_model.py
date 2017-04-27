import pandas as pd
import itertools
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import recall_score

class svm:
    """
    this is svm process for classifier multi-labels,
    from csv load data as pd data frame and clean.
    process_Y use MultilabelBinarizer to convert tags as sparse matrix
    :param min_number is the min times the tag appear, use this param to filter the imbalance tags
    process_X use TfidfVectorize function to convert text to vector, in this class I choose body as learing text.
    """

    @classmethod
    def from_csv(pd, file_path, sep = ',', quote_character='|'):
        with open(file_path, 'rU') as infile:
            data = pd.read_csv(infile,  quotechar=quote_character,header = None, sep = sep)
            data.column = ['title','body','tags']
            all_tags = data.tags.apply(lambda x: x.replace('<', '').split('>')[:-1])
            data['tags'] = all_tags
            return pd(data = data)


    def process_Y(self, data, min_number):
        data_tags = data['tag']
        data_tags = [x for x in list(itertools.chain(*data_tags)) if x]
        ct = Counter(data_tags)
        vocab = ct.most_common()
        pd_tag = pd.DataFrame(vocab)
        pd_tag.columns = ['name', 'count']
        tags_list = pd_tag[pd_tag['count'] >= min_number]
        tag_used = [x for x in tags_list.name]
        tag_used
        tag_result = []
        for tags in data_tags:
            tag_result.append([x for x in tags if x in tag_used])
        tag_result
        mlb = MultiLabelBinarizer()
        Y_train = mlb.fit_transform(tag_result)
        return Y_train

    def Preprocess(text):
        return text.lower()

    def Token(text):
        tokens = word_tokenize(text)
        stemmer = SnowballStemmer('english')
        stop = stopwords.words('english')
        stop.extend(['want', 'to', "'s"])
        english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', "'",
                                "'m", "-", "n't"]
        stop.extend(english_punctuations)
        to_be_removed = ["''", '``', '...', "'d", '2', '--', '/', '>', '3', '=', 'x', 'e.g', '2005', '2008', '{', '}',
                         '2.0', '0']
        stop.extend(to_be_removed)
        texts_filtered = [stemmer.stem(word) for word in tokens]
        final_res = [word for word in texts_filtered if not word in stop]
        return final_res

    def process_X(self, data, max_features):
        vectorizer = TfidfVectorizer(preprocessor=Preprocess, tokenizer=Token, max_features=max_features)
        X_train = vectorizer.fit_transform(data.body)
        return X_train


    def classifier(self,data):
        """
        OneVsRestclassifer is a classifer used for class multi_label,
        use SVM model with liner kernel to handle sparse matrix.
        recall_score return how the model fit the data.
        """
        X_train = prpcess_X(data)
        Y_train = process_Y(data)
        clf = OneVsRestClassifier(SVC(C=0.025, class_weight='balanced', kernel='linear'))
        clf.fit(X_train, Y_train)
        pre = clf.predict(X_test)
        accurate = recall_score(Y_test, pre, average='weighted')
        result = mlb.inverse_transform(pre)
        data['tag']=pd.Series(result)
        return data,accurate

