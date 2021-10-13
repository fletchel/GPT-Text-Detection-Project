# logistic regression baseline using word2vec

import gensim
import pandas as pd
import nltk
import pickle
import sys
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split



file_names = ["CNNArticles.csv", "fb_edit.csv", "ireland_headlines.csv", "tweets_edited.csv"]
temps = ["0_5", "0_75", "1_0", "1_25", "1_5"]
temps = ["1_0"]

#data_, type_, temp_ = sys.argv[1:4]

with open("models/LR.pickle", "rb") as f:

    LR = pickle.load(f)

with open("models/LR_test_data.pickle", "rb") as f:

    test_data = pickle.load(f)


test_data_2 = {} # test_data dictionary w/ file names/temperatures as keys

for f in file_names:

    test_data_2[f] = [test_data[f], list(np.zeros(len(test_data[f])))]

    for t in temps:

        test_data_2[f][0] = test_data_2[f][0] + test_data[(f, t)]
        test_data_2[f][1] = test_data_2[f][1] + list(np.ones(len(test_data[(f,t)])))


for k in test_data_2.keys():

    x = test_data_2[k][0]
    y = test_data_2[k][1]

    acc = LR.score(x, y)
    print(k)
    print(acc)
    print("\n")


for k in test_data.keys():

    x = test_data[k]

    if type(k) == tuple:

        y = np.ones(len(x))

    else:

        y = np.zeros(len(x))

        p = [z for i in x for z in i]

    acc = LR.score(x, y)
    print(k)
    print(acc)
    print("\n")



