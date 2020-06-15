import os
import pandas as pd
import numpy as np

os.chdir("C:\\Users\\user\\Documents\\Python\\Heroku-Demo-master\\spam")

messages = pd.read_csv("Spam SMS Collection",sep = '\t',
                      names = ['label','message'])

import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re

ps = PorterStemmer()
Corpus = []

for i in range(len(messages['message'])):
    review = re.sub('[^a-zA-Z]',' ',messages['message'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(words) for words in review if words not in set(stopwords.words('english'))]
    review = ' '.join(review)
    Corpus.append(review)

from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.naive_bayes import MultinomialNB

tf = TfidfVectorizer(max_features = 5000)
X = tf.fit_transform(Corpus).toarray()
X = pd.DataFrame(X)

import pickle

pickle.dump(tf,open('Tfid.pkl','wb'))

Y = messages['label']

FullRaw = pd.concat([X,Y], axis =1)

from sklearn.model_selection import train_test_split

Train,Test = train_test_split(FullRaw,test_size = 0.3, random_state =123)

Train_X = Train.drop(['label'], axis =1)
Train_Y = Train['label'].copy()
Test_X = Test.drop(['label'], axis =1)
Test_Y = Test['label'].copy()

Model = MultinomialNB().fit(Train_X,Train_Y)

Test_pred = Model.predict(Test_X)

from sklearn.metrics import confusion_matrix

Con_Mat = confusion_matrix(Test_pred,Test_Y)

sum(np.diag(Con_Mat))/Test_Y.shape[0]*100


 
import pickle

""" model pickled"""
pickle.dump(Model,open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))


""" Tf pickled""" 

pickle.dump(tf,open('Tfid.pkl','wb'))
Tf_vect = pickle.load(open('Tfid.pkl','rb'))

