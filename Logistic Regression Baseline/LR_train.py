import pandas as pd
import nltk
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import pickle

file_names = ["CNNArticles", "fb_edit", "ireland_headlines", "tweets_edited"]
temps = ["0_5", "0_75", "1_0", "1_25", "1_5"]


train_data = pickle.load(open("data/embedded_data/embedded_train.pkl", "rb"))
test_data = pickle.load(open("data/embedded_data/embedded_test.pkl", "rb"))

x = train_data
y = list(np.ones(57591)) + list(np.zeros(57600))
num_nan = 0

for i, e in enumerate(x):

    if np.any(np.isnan(e)):

        x[i] = np.zeros(300)


x_test = []
y_test = []

print(len(x))
print(len(y))

for f in file_names:

    x_test = x_test + test_data[f][0]
    y_test = y_test + list(np.zeros(len(test_data[f][0])))

    print(len(test_data[f][0]))
    print(f + " \n")

    for t in temps:

        x_test = x_test + test_data[(f,t)][0]
        y_test = y_test + list(np.ones(len(test_data[(f,t)][0])))
        print(len(test_data[(f,t)][0]))
        print(f + " " + t + " \n")


for i, e in enumerate(x_test):

    if np.any(np.isnan(e)):

        x_test[i] = np.zeros(300)

for k in test_data.keys():

    for i, e in enumerate(test_data[k][0]):

        if np.any(np.isnan(e)):

            test_data[k][0][i] = np.zeros(300)

LR = LogisticRegression(max_iter=8000)

LR.fit(x, y)

with open("models/final_LR.pkl", "wb") as f:

    pickle.dump(LR, f)


