import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import pickle

test_proportion = 0.2


file_names = ["CNNArticles.csv", "fb_edit.csv", "ireland_headlines.csv", "tweets_edited.csv"]
#file_names = ["ireland_headlines.csv"]
temps = ["0_5", "0_75", "1_0", "1_25", "1_5"]

gpt_vectors = []
human_vectors = []

test_data = {}


for f in file_names:

    for t in temps:

        path = "data/embedded_training_data/gpt/embedded_" + f.replace('.csv', '') + t + '.csv'

        vectors = pd.read_csv(path).to_numpy()

        if t in ["0_5", "0_75", "1_25", "1_5"]:
            indices = np.random.choice(len(vectors), size=min(2000, len(vectors)), replace=False)

        else:
            indices = np.random.choice(len(vectors), size=min(10000, len(vectors)), replace=False)

        vectors = vectors[indices, :]

        vectors, test_vectors = train_test_split(vectors, test_size=test_proportion)

        for v in vectors:

            gpt_vectors.append(v)

        test_data[(f, t)] = list(test_vectors)


for f in file_names:

    path = "data/embedded_training_data/human/embedded_" + f

    vectors = pd.read_csv(path).to_numpy()



    indices = np.random.choice(vectors.shape[0], size=18000, replace=False)

    vectors = vectors[indices, :]

    vectors, test_vectors = train_test_split(vectors, test_size=test_proportion)

    count = 0

    for v in vectors:

        if not np.any(np.isnan(v)):
            human_vectors.append(v)

        else:
            count = count+1

    f_test_vectors = []

    for v in test_vectors:

        if not np.any(np.isnan(v)):
            f_test_vectors.append(v)

    test_data[f] = list(f_test_vectors)



    print(f)
    print(count)

print(len(gpt_vectors))
print(len(human_vectors))

x = gpt_vectors + human_vectors
y = [1]*len(gpt_vectors) + [0]*len(human_vectors)


LR = LogisticRegression(max_iter=8000)

LR.fit(x, y)

print(LR.score(x, y))

with open("models/LR.pickle", "wb") as f:

    pickle.dump(LR, f)

print(test_data.keys())

with open("models/LR_test_data.pickle", "wb") as f:

    pickle.dump(test_data, f)
