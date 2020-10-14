# import the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import libraries for nlp
import nltk
import string
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# nltk.download('punkt')

# REAd the dataset
data = pd.read_csv('dataset//train_E6oV3lV.csv')

training, test = train_test_split(data, test_size=.33, random_state=42)
train_x = training['tweet']
train_y = training['label']
test_x = test['tweet']
test_y = test['label']

# vectorizer = CountVectorizer()
vectorizer = TfidfVectorizer()

#vectorizer.fit(train_x)
#train_x_vector = vectorizer.transform(train_x)
train_x_vector = vectorizer.fit_transform(train_x)
test_x_vector = vectorizer.transform(test_x)

print(train_x[0])
print(train_x_vector[0].toarray())

from sklearn import svm

clf_svm =svm.SVC(kernel='linear')
clf_svm.fit(train_x_vector, train_y)

clf_svm.predict(test_x_vector[0])

# mean accuracy
print(clf_svm.score(test_x_vector, test_y))

test_dataset = pd.read_csv('dataset//test_tweets_anuFYb8.csv')

test_dataset_vector = vectorizer.transform(test_dataset['tweet'])

labels = clf_svm.predict(test_dataset_vector)
sample_submission = pd.read_csv('dataset//sample_submission_gfvA5FD.csv')
sample_submission['label'] = labels

# write mode
sample_submission = pd.DataFrame(sample_submission)
sample_submission.to_csv('sample_submission.csv')