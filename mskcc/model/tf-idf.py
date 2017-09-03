# encoding=utf-8

import sys
import pandas as pd
import random
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import SVC

TEST = False

# read data
train_text = sys.argv[1]
train_variant = sys.argv[2]

if TEST:
    test_text = sys.argv[3]
    test_samples = pd.read_csv(test_text, sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])

text = pd.read_csv(train_text, sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
variant = pd.read_csv(train_variant, engine='python')
samples = text.merge(variant, on='ID')
samples.drop('ID', axis=1)



if TEST:
    train_samples = samples
    print 'train sample count:', len(train_samples)

else:
    # split into train and validate set
    indices = range(0, len(samples))
    fx = 0.9
    train_count = int(len(samples)*fx)

    random.seed(40)
    random.shuffle(indices)
    random.seed()

    train_indices = indices[:train_count]
    validate_indices = indices[train_count:]

    train_samples = samples.iloc[train_indices]
    validate_samples = samples.iloc[validate_indices]

    print "train sample count {}, validate sample count {}".format(len(train_samples), len(validate_samples))

# get tf-idf vector
print 'fitting train samples'
count_vect = CountVectorizer(stop_words='english', max_df=1.0)
count_vect.fit(train_samples['Text'])
print count_vect.get_feature_names()
print count_vect.get_stop_words()
X_train_counts = count_vect.transform(train_samples['Text'])
if TEST:
    X_test_counts = count_vect.transform(test_samples['Text'])
else:
    X_validate_counts = count_vect.transform(validate_samples['Text'])

tfidf_transformer = TfidfTransformer()
tfidf_transformer.fit(X_train_counts)
X_train_tfidf = tfidf_transformer.transform(X_train_counts)
if TEST:
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)
else:
    X_validate_tfidf = tfidf_transformer.transform(X_validate_counts)

print 'feature size {}'.format(X_train_tfidf.shape)

Y_train = train_samples['Class'].values - 1
if not TEST:
    Y_validate = validate_samples['Class'].values - 1
    Y_validate_one_hot = np.eye(9)[Y_validate]

# train
print 'trainging...'
clf = SVC(kernel='linear', probability=True)
clf.fit(X_train_tfidf, Y_train)

if TEST:
    print 'testing...'
    proba = clf.predict_proba(X_test_tfidf)
    submission = pd.DataFrame(proba, columns=['class' + str(c + 1) for c in range(9)])
    submission['ID'] = test_samples['ID'].values
    submission.to_csv('submission.csv', index=False)

else:
    # evaluate on validate set
    print 'validating...'
    acc = clf.score(X_validate_tfidf, Y_validate)
    proba = clf.predict_proba(X_validate_tfidf)
    multi_log_loss = -np.mean(np.sum(np.multiply(Y_validate_one_hot,np.log(proba)), axis=1))
    print 'Validate set multipy log loss: {:.8f}, accuracy: {}'.format(multi_log_loss, acc)
