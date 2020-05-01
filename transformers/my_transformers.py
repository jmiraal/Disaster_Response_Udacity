import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

import nltk
from nltk.corpus import stopwords
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
nltk.download('maxent_ne_chunker')
nltk.download('words')

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer



class GPEExtractor(BaseEstimator, TransformerMixin):

    '''
       Tranformer to detect if ther is a GPE into the message
    '''

    def gpe_extractor(self, text):
        # tokenize by sentences
        sentence_list = nltk.sent_tokenize(text)
        
        for sentence in sentence_list:
            # tokenize each sentence into words and tag part of speech
            tree = nltk.ne_chunk(nltk.pos_tag(word_tokenize(sentence)))
            if ('GPE' in str(tree)):
                
                return True
            else:
                return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        # apply gpe_extractor function to all values in X
        X_tagged = pd.Series(X).apply(self.gpe_extractor)
        return pd.DataFrame(X_tagged)
  
  
class PersonExtractor(BaseEstimator, TransformerMixin):

    '''
       Tranformer to detect if a person in named into the message
    '''

    def person_extractor(self, text):
        # tokenize by sentences
        sentence_list = nltk.sent_tokenize(text)
        
        for sentence in sentence_list:
            # tokenize each sentence into words and tag part of speech
            tree = nltk.ne_chunk(nltk.pos_tag(word_tokenize(sentence)))
            if ('PERSON' in str(tree)):
                return True
            else:
                
                return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        # apply person_extractor function to all values in X
        X_tagged = pd.Series(X).apply(self.person_extractor)
        return pd.DataFrame(X_tagged)


class OrganizationExtractor(BaseEstimator, TransformerMixin):
    '''
       Tranformer to detect if there is an Organization in the message
    '''

    def organization_extractor(self, text):
        # tokenize by sentences
        sentence_list = nltk.sent_tokenize(text)
        
        for sentence in sentence_list:
            # tokenize each sentence into words and tag part of speech
            tree = nltk.ne_chunk(nltk.pos_tag(word_tokenize(sentence)))
            if ('ORGANIZATION' in str(tree)):
                return True
            else:
                return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        # apply organization_extractor function to all values in X
        X_tagged = pd.Series(X).apply(self.organization_extractor)
        return pd.DataFrame(X_tagged)
        
        

class Reshape_Array(BaseEstimator, TransformerMixin):
    '''
        It converts a Series or a Dataframe into a matrix with two dimensions.
    '''
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.values.reshape(-1, 1)


class ColumnSelector(BaseEstimator, TransformerMixin):
    '''
       Transformer to select a column where to  apply the transformation
    '''

    def __init__(self, column):
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.column]


