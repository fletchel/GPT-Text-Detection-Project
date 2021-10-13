# convert raw training and test data into embeddings

import pandas as pd
import gensim
import nltk
import numpy as np
import pickle

embeddings_path = "embeddings/glove_vectors.txt"

gpt_text = list(pd.read_csv("data/raw_data/train_data/gpt_text.csv").iloc[:, 0])
human_text = list(pd.read_csv("data/raw_data/train_data/human_text.csv").iloc[:, 0])

training_text = gpt_text + human_text
print(len(training_text))

training_text = [str(x) for x in training_text]

with open("data/raw_data/test_data/test_data.pkl", "rb") as f:

    test_data = pickle.load(f)

model = gensim.models.KeyedVectors.load_word2vec_format(embeddings_path, binary=False, limit=30000)
tokenizer = nltk.TweetTokenizer(strip_handles=True)

train_embeddings = []

gpt_train = 0
human_train = 0

for i, t in enumerate(training_text):

    if len(t) > 2:

        tokenized = tokenizer.tokenize(t)

        if tokenized[-1:] == '"':
            tokenized = tokenized[:-1]

        if len(tokenized) > 0:
            if tokenized[0] == '"':
                tokenized = tokenized[1:]

        if "<" in tokenized:
            tokenized.remove("<")

        if "|startoftext|" in tokenized:
            tokenized.remove("|startoftext|")

        if ">" in tokenized:
            tokenized.remove(">")

        for k, w in enumerate(tokenized):

            if w[-1:] == ".":
                tokenized[k] = w[:-1]

    else:
        tokenized = ' '

    vector = np.zeros(300)

    for token in tokenized:

        try:

            vector = vector + model[token]

        except:

            pass

    vector = vector / len(tokenized)

    if not vector.any():
        # print("fail")
        pass


    train_embeddings.append(vector)

test_embeddings = {}

for k in test_data.keys():

    test_embeddings[k] = [[],[]]

    for i, text in enumerate(test_data[k].iloc[:, 0]):

        t = str(text)
        if len(t) > 2:

            tokenized = tokenizer.tokenize(t)

            if tokenized[-1:] == '"':
                tokenized = tokenized[:-1]

            if len(tokenized) > 0:
                if tokenized[0] == '"':
                    tokenized = tokenized[1:]

            if "<" in tokenized:
                tokenized.remove("<")

            if "|startoftext|" in tokenized:
                tokenized.remove("|startoftext|")

            if ">" in tokenized:
                tokenized.remove(">")

            for h, w in enumerate(tokenized):

                if w[-1:] == ".":
                    tokenized[h] = w[:-1]

        else:
            tokenized = ''

        vector = np.zeros(300)

        for token in tokenized:


            try:

                vector = vector + model[token]

            except:

                pass

        vector = vector / len(tokenized)

        if not vector.any():
            # print("fail")
            pass

        test_embeddings[k][0].append(vector)
        test_embeddings[k][1].append(len(tokenized))

with open("data/embedded_data/embedded_train.pkl", "wb") as f:

    pickle.dump(train_embeddings, f)

with open("data/embedded_data/embedded_test.pkl", "wb") as f:

    pickle.dump(test_embeddings, f)

