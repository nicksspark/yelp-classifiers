# LSTM with Dropout for sequence classification in the IMDB dataset
import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split

# fix random seed for reproducibility
numpy.random.seed(0)
# load the dataset but only keep the top n words, zero the rest
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
# truncate and pad input sequences
max_review_length = 500

X = []
y = []
with open('./sentiment labelled sentences/yelp_labelled.txt') as fp:
    for line in fp:
        l = line.split('\t')
        X.append(l[0])
        y.append(int(l[1].replace('\n', '')))

vec = TfidfVectorizer()
X_vec = vec.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, random_state=5)

# X_train = sequence.pad_sequences(X_train.shape[0], maxlen=max_review_length)
# X_test = sequence.pad_sequences(X_test.shape[0], maxlen=max_review_length)

# create the model
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=2035))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, epochs=10, batch_size=16)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
