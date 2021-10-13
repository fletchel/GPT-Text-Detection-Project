import pickle
import pandas as pd
import nltk
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

file_names = ["CNNArticles", "fb_edit", "ireland_headlines", "tweets_edited"]
temps = ["0_5", "0_75", "1_0", "1_25", "1_5"]

with open("models/final_LR.pkl", "rb") as f:

    LR = pickle.load(f)

test_data = pickle.load(open("data/embedded_data/embedded_test.pkl", "rb"))

for k in test_data.keys():

    for i, e in enumerate(test_data[k][0]):

        if np.any(np.isnan(e)):

            test_data[k][0][i] = np.zeros(300)

for f in file_names:

    x = test_data[f][0]
    y = list(np.zeros(len(test_data[f][0])))

    for t in temps:

        x = x + test_data[(f,t)][0]
        y = y + list(np.ones(len(test_data[(f, t)][0])))

    print(LR.score(x, y))
    print(f)

test_x = []
test_y = []

for t in temps:

    test_x = []
    test_y = []

    for f in file_names:

        test_x = test_x + test_data[(f,t)][0]
        test_y = test_y + list(np.ones(len(test_data[(f,t)][0])))

    print(LR.score(test_x, test_y))
    print(t)
    print("\n")


test_x = []
test_y = []
x_length = []

for f in file_names:

    test_x = test_x + test_data[f][0]
    x_length = x_length + test_data[f][1]

    test_y = test_y + list(np.zeros(len(test_data[f][0])))

    for t in temps:

        test_x = test_x + test_data[(f,t)][0]
        x_length = x_length + test_data[(f,t)][1]

        test_y = test_y + list(np.ones(len(test_data[(f,t)][0])))


    # now we have x, y and lengths of x

    q1 = np.percentile(x_length, 25)
    q2 = np.percentile(x_length, 50)
    q3 = np.percentile(x_length, 75)

    x1 = [x for i, x in enumerate(test_x) if x_length[i] <= q1]
    x2 = [x for i, x in enumerate(test_x) if q1 < x_length[i] <= q2]
    x3 = [x for i, x in enumerate(test_x) if q2 < x_length[i] <= q3]
    x4 = [x for i, x in enumerate(test_x) if x_length[i] > q3]

    y1 = [y for i, y in enumerate(test_y) if x_length[i] <= q1]
    y2 = [y for i, y in enumerate(test_y) if q1 < x_length[i] <= q2]
    y3 = [y for i, y in enumerate(test_y) if q2 < x_length[i] <= q3]
    y4 = [y for i, y in enumerate(test_y) if x_length[i] > q3]

    print(LR.score(x1, y1))
    print(q1)
    print(LR.score(x2, y2))
    print(q2)
    print(LR.score(x3, y3))
    print(q3)
    print(LR.score(x4, y4))

    print(f)
    print("\n")





