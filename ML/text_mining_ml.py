import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
wnl = WordNetLemmatizer()
class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [wnl.lemmatize(wnl.lemmatize(t,'v'),'n') for t in word_tokenize(doc)]

# Import the relevant scikit packages
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

data_path='/content/overview-of-recordings.csv'
df=pd.read_csv(data_path)

print(df.info(),len(df))

#To get The Transcripted Phrases with their prompt [TARGET] :
df[['phrase','prompt']]

# Unique Classes Category :
df['prompt'].unique()
print("Total Unique Classes : ")

#Total Unique Classes : 
'''
(array(['Emotional pain', 'Hair falling out', 'Heart hurts',
        'Infected wound', 'Foot ache', 'Shoulder pain',
        'Injury from sports', 'Skin issue', 'Stomach ache', 'Knee pain',
        'Joint pain', 'Hard to breath', 'Head ache', 'Body feels weak',
        'Feeling dizzy', 'Back pain', 'Open wound', 'Internal pain',
        'Blurry vision', 'Acne', 'Muscle pain', 'Neck pain', 'Cough',
        'Ear ache', 'Feeling cold'], dtype=object)
'''

#Check and Remove Duplicates :

df.drop_duplicates(subset='phrase',inplace=True)
df = df.reset_index()

# select random sentence
sentence = df.sample(n=1)['phrase'].values[0]
# create tokens from sentence
tokens = nltk.word_tokenize(sentence)
# create tagged sentence
tagged = nltk.pos_tag(tokens)


# Get Insights From Data using CountVectoriser and Bag of words
documents = df['phrase'].values
count_vectorizer = CountVectorizer()
bag_of_words = count_vectorizer.fit_transform(documents)
feature_names = count_vectorizer.get_feature_names()
feature_names,len(feature_names)

pd.DataFrame(bag_of_words.toarray(), columns = feature_names).head(n=6)

#Model Pipeline to get class classifications:

from sklearn import preprocessing
le = preprocessing.LabelEncoder()

# generate data
target = le.fit_transform(df['prompt'].values)
data = df['phrase'].values
classes = le.classes_

# print results
print('# of classes: {}'.format(len(classes)))
print('Fitted classes:')
print(classes)



#Train Test Split Normal Method :
X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=0.10, random_state=42)
print('Training data size: {}'.format(len(X_train)))
print('Testing data size: {}'.format(len(X_test)))

plt.figure(figsize=(10,6))
width = 0.35
unique, counts = np.unique(y_test, return_counts=True)
plt.bar(unique,counts/np.sum(counts),width,label='test')

unique, counts = np.unique(y_train, return_counts=True)
plt.bar(unique+width,counts/np.sum(counts),width,label='train')

plt.xlabel('Class'); plt.ylabel('Occurance'); plt.legend()


#Balance the class imbalance occurance of each class using stratify parameter :

X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=0.10, stratify=target,random_state=42)
print('Training data size: {}'.format(len(X_train)))
print('Testing data size: {}'.format(len(X_test)))
plt.figure(figsize=(10,6))
width = 0.35
unique, counts = np.unique(y_test, return_counts=True)
plt.bar(unique,counts/np.sum(counts),width,label='test')

unique, counts = np.unique(y_train, return_counts=True)
plt.bar(unique+width,counts/np.sum(counts),width,label='train')

plt.xlabel('Class'); plt.ylabel('Occurance'); plt.legend()


#  Using Resampled Accuracy for Bootstrap confidence intervals for test accuracy:

def print_resampled_accuracy(predicted,y,samples=1000):

    from sklearn.utils import resample
    acc = []
    for _ in range(samples):
        r_p, r_y = resample(predicted, y)
        acc.append(np.mean(r_p == r_y))

    lc,m,uc = np.percentile(acc,[2.5,50.,97.5])
    print('Accuracy : {:.2%} ({:.2%} - {:.2%})'.format(m,lc,uc))


#Creating Pipeline : 

#1st Classifier Without Lemmatizer-> Multinomial NB
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])

text_clf.fit(X_train, y_train)
predicted = text_clf.predict(X_test)
val_acc = np.mean(predicted == y_test) 
print('Validation accuracy: {:.2%}'.format(val_acc))

text_lemma_clf = Pipeline([
    ('vect', CountVectorizer(tokenizer=LemmaTokenizer())),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])

text_lemma_clf.fit(X_train, y_train)
predicted = text_lemma_clf.predict(X_test)
 

#Second Classifier : Linear SVC :
text_sgd_clf = Pipeline([
    ('vect', CountVectorizer(tokenizer=LemmaTokenizer())),
    ('tfidf', TfidfTransformer()),
    ('clf', LinearSVC())        
    ])

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC

print('Training data Results')
predicted = text_sgd_clf.predict(X_train)
print_resampled_accuracy(predicted,y_train,samples=1200)


print('Test data Results')
predicted = text_sgd_clf.predict(X_test)
print_resampled_accuracy(predicted,y_test,samples=1200)


text_rf_clf = Pipeline([
    ('vect', CountVectorizer(tokenizer=LemmaTokenizer())),
    ('tfidf', TfidfTransformer()),
    ('clf', RandomForestClassifier(n_estimators=1400,max_depth=32,max_features='auto',criterion='gini')),

])

text_rf_clf.fit(X_train, y_train)

print('Training data Results:')
predicted = text_rf_clf.predict(X_train)
print_resampled_accuracy(predicted,y_train,samples=1200)


print('Test data Results:')
predicted = text_rf_clf.predict(X_test)
print_resampled_accuracy(predicted,y_test,samples=1200)

print(classification_report(y_test, predicted, target_names=classes))
