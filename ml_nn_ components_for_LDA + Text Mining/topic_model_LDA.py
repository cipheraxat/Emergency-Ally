# LDA Topic Modelling:
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer,TfidfTransformer
from sklearn.decomposition import  LatentDirichletAllocation

tfidf=TfidfTransformer()

from sklearn import preprocessing
le = preprocessing.LabelEncoder()

def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic {:d}:".format(topic_idx))
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))


data_path='/content/overview-of-recordings.csv'
df=pd.read_csv(data_path)
target = le.fit_transform(df['prompt'].values)
data = df['phrase'].values

no_features = 1400

# Using LDA with CountVectoriser
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
tf = tf_vectorizer.fit_transform(data)
tf_feature_names = tf_vectorizer.get_feature_names()

n_components = 5


# Run LDA
lda = LatentDirichletAllocation(n_components=n_components, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)

no_top_words = 10

print('LDA Topics\n----------')
display_topics(lda, tf_feature_names, no_top_words)



output = lda.transform(tfidf)


for topic in range(output.shape[1]):
    print('Topics Predicted By LDA No.{}:'.format(topic))
    inds = np.argsort(output[:,topic])[::-1]
    for ind in inds[10:15]:
        print(data[ind])
    print('\n')