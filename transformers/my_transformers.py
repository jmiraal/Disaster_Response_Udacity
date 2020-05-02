import re
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


class Tokenizer(BaseEstimator, TransformerMixin):
    '''
       Tranformer to apply tokenization. We need include the function tokenize
       into a class to be called inside the Heroku platoform. Then we implement
       this tokeniztion inside the Pipeline in this way:
       
       (train_classifier.py)
       .
       .
       from transformers.my_transformers import 
       .
       .
       .
       ('vect', CountVectorizer(tokenizer=Tokenizer.tokenize))
       
       As a local script we could have usee the function directly in the 
       Pipeline, without the necessity of included it inside an class.
       
       ('vect', CountVectorizer(tokenizer=tokenize))
       
    '''
    def tokenize(text):
        '''
        USAGE 
           clean and tokenize a message
        INPUT
           text: String we want to clean and tokenize       
        OUTPUT
           clean_tokens: list of tokens         
        '''
        # patterns to detect url'sand users
        url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        #hashtag_regex = '#[A-Za-z0-9]*'
        user_regex = '(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z]+[A-Za-z0-9-_]+)'
    
        # change urls by the word "urlplaceholder"
        detected_urls = re.findall(url_regex, text)
        for url in detected_urls:
            text = text.replace(url, "urlplaceholder")
    
        detected_users = re.findall(user_regex, text)
        for users in detected_users:
            text = text.replace(users, "userplaceholder")
    
    
        # replace all the chars not in a-z, A-Z or 0-9 by " "
        text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    
        # tokenize the text and drop the the stop words in Enghish
        tokens = word_tokenize(text)
        tokens = [w for w in tokens if w not in stopwords.words("english")]
    
        # initialize the lemmatizer
        lemmatizer = WordNetLemmatizer()

        # lemmatize each token and return a list of lower lematized tokens
        clean_tokens = []
        for tok in tokens:
            clean_tok = lemmatizer.lemmatize(tok).lower().strip()
            clean_tokens.append(clean_tok)

        return clean_tokens


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


