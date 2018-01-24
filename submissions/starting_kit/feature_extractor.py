from __future__ import unicode_literals
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from keras.preprocessing.text import Tokenizer
import unicodedata
import pandas as pd
import numpy as np
import string
import re
from nltk.corpus import stopwords

stpwrds = set([stopword for stopword in stopwords.words('english')])
stpwrds.update({'admission', 'birth', 'date', 'discharge', 'service','sex'})
punct = set(string.punctuation.replace('-', ''))
punct.update(["``", "`", "..."])

def clean_text_simple(text, my_stopwords=stpwrds, punct=punct, remove_stopwords=True, stemming=False):
    text = text.lower()
    text = ''.join(l for l in text if l not in punct) # remove punctuation (preserving intra-word dashes)
    text = re.sub(' +',' ',text) # strip extra white space
    text = text.strip() # strip leading and trailing white space 
    tokens = text.split() # tokenize (split based on whitespace)
    tokens = [w for w in tokens if w.isalpha()]
    tokens = [w for w in tokens if len(w) > 2]

    if remove_stopwords:
        # remove stopwords from 'tokens'
        tokens = [x for x in tokens if x not in my_stopwords]

    if stemming:
        # apply stemmer
        stemmer = SnowballStemmer('english')
        tokens = [stemmer.stem(t) for t in tokens]

    return tokens



def document_preprocessor(doc):
    # TODO: is there a way to avoid these encode/decode calls?
    try:
        doc = unicode(doc, 'utf-8')
    except NameError:  # unicode is a default on python 3
        pass
    doc = unicodedata.normalize('NFD', doc)
    doc = doc.encode('ascii', 'ignore')
    doc = doc.decode("utf-8")
    return str(doc)


from nltk.stem import SnowballStemmer

stemmer = SnowballStemmer('english')

def token_processor(tokens):
    for token in tokens:
        #remove special chars
        token=''.join(e for e in token if e.isalnum())
        yield stemmer.stem(token)

class FeatureExtractor(TfidfVectorizer):
    """Convert a collection of raw docs to a matrix of TF-IDF features. """

    def __init__(self):
        self.min_occur = 1
        self.max_length = -1
        self.vocab_size = -1
        self.tokenizer = Tokenizer() # create the tokenizer
         
        super(FeatureExtractor, self).__init__(
                analyzer='word',stop_words ='english', preprocessor=document_preprocessor)

    def fit(self, X_df, y=None):
        """Learn a vocabulary dictionary of all tokens in the raw documents.

        Parameters
        ----------
        X_df : pandas.DataFrame
            a DataFrame, where the text data is stored in the ``TEXT``
            column.
        """
        
        super(FeatureExtractor, self).fit(X_df.TEXT)
        
        
        return self

    def fit_transform(self, X_df, y=None):
        self.fit(X_df)
        return self.transform(self.X_df)

    def transform(self, X_df):

        statements = pd.Series(X_df.TEXT).apply(clean_text_simple)

        vocab = Counter()
        for statement in statements:
            vocab.update(statement)
        tokens = [k for k,c in vocab.items() if c >= self.min_occur]
        statements = statements.apply(lambda x: [w for w in x if w in tokens])
        statements = statements.apply(lambda x: ' '.join(x))
        statements = list(statements.values)
        vec_c = TfidfVectorizer(ngram_range=(1, 1))
        tf_idf = vec_c.fit_transform(statements)
        return tf_idf

    def build_tokenizer(self):
        """
        Internal function, needed to plug-in the token processor, cf.
        http://scikit-learn.org/stable/modules/feature_extraction.html#customizing-the-vectorizer-classes
        """
        tokenize = super(FeatureExtractor, self).build_tokenizer()
        return lambda doc: list(token_processor(tokenize(doc)))
