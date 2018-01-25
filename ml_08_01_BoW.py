import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

count = CountVectorizer()
docs = np.array(['The sun is shining',
                 'The weathere is sweet',
                 'The sun is shining, the weather is sweet, and one and one is tow'])

bag = count.fit_transform(docs)

print(count.vocabulary_)
print(bag.toarray())

tfidf = TfidfTransformer(use_idf=True, norm='l2', smooth_idf=True)
np.set_printoptions(precision=2)
print(tfidf.fit_transform(count.fit_transform(docs)).toarray())

