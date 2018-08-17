from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, make_scorer, classification_report
import pandas as pd

# set = pd.read_csv('./sentiment labelled sentences/yelp_labelled.txt', sep='\t', names=['data', 'target'])
# print('set', set.head())
# vectorizer = CountVectorizer()
# v_data = vectorizer.fit_transform(set['data'])
# X_train, X_test, y_train, y_test = train_test_split(v_data, set.target, test_size=0.25, random_state=1)
#
# a = [0.0]
# base = 0.005
# for i in range(100):
#     a.append(base)
#     base += 0.005
#
# a = [0, .5, 1, 1.5, 2]
# params = {'alpha': a}
#
# model = MultinomialNB()
# scorer = make_scorer(f1_score, average='micro')
# model = GridSearchCV(model, params, scoring=scorer)
# prediction = model.fit(X_train, y_train).predict(X_test)
# print(classification_report(y_train, model.predict(X_train)))
# print(classification_report(y_test, prediction))

X = []
y = []
with open('./sentiment labelled sentences/yelp_labelled.txt') as fp:
    for line in fp:
        l = line.split('\t')
        X.append(l[0])
        y.append(int(l[1].replace('\n', '')))

vecs = [
    CountVectorizer(binary=False),
    CountVectorizer(binary=False, ngram_range=(1,3)),
    CountVectorizer(binary=True),
    CountVectorizer(binary=True, ngram_range=(1,3)),
    TfidfVectorizer(),
    TfidfVectorizer(ngram_range=(1, 3)),
    TfidfVectorizer(sublinear_tf=True),
    TfidfVectorizer(ngram_range=(1, 3), sublinear_tf=True),
]
classifiers = [
    BernoulliNB(),
    MultinomialNB(),
    XGBClassifier()
]

diff = 100
final = None

for vec in vecs:
    print(vec)
    X_vec = vec.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, random_state=5)
    for clf in classifiers:
        print(clf)
        clf.fit(X_train, y_train)
        train_score = f1_score(y_train, clf.predict(X_train))
        test_score = f1_score(y_test, clf.predict(X_test))
        if abs(test_score - train_score) < diff:
            diff = abs(test_score - train_score)
            final = (clf, vec)
        print('train:', {train_score}, 'test:', {test_score})

print('FINAL', diff, final)
