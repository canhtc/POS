from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam
from keras.layers import Dense, LSTM, InputLayer, Bidirectional, TimeDistributed, Embedding, Activation
from keras.models import Sequential
from sklearn.model_selection import train_test_split
import numpy as np
from pathlib import Path
from tensorflow import keras
import tensorflow as tf
# Read data
data_folder = Path("data/")
train_data = data_folder/"pos-train"
train_data = open(train_data, "r").readlines()
train_data = [t.split() for t in train_data]

# # Clean data
tagged_sentences = []
for data in train_data:
    childs = []
    for t in data:
        child = t.strip().split("/")
        if len(child) == 2 and child[1].isalpha():
            childs.append(tuple(child))
        else:
            print("Somethings wrong!!!")
    if(len(childs) > 0):
        tagged_sentences.append(childs)


# # Restructure the data. Separate the words from the tags.
sentences, sentence_tags = [], []
for tagged_sentence in tagged_sentences:
    sentence, tags = zip(*tagged_sentence)
    sentences.append(np.array(sentence))
    sentence_tags.append(np.array(tags))

# print(len(tagged_sentences))
# print(count)
# print(sentence_tags)

(train_sentences,
 test_sentences,
 train_tags,
 test_tags) = train_test_split(sentences, sentence_tags, test_size=0.2)

# #
words, tags = set([]), set([])

for s in train_sentences:
    for w in s:
        words.add(w.lower())

for ts in train_tags:
    for t in ts:
        tags.add(t)

word2index = {w: i + 2 for i, w in enumerate(list(words))}
word2index['-PAD-'] = 0  # The special value used for padding
word2index['-OOV-'] = 1  # The special value used for OOVs

tag2index = {t: i + 1 for i, t in enumerate(list(tags))}
tag2index['-PAD-'] = 0  # The special value used to padding


# Let’s now convert the word dataset to integer dataset, both the words and the tags
train_sentences_X, test_sentences_X, train_tags_y, test_tags_y = [], [], [], []

for s in train_sentences:
    s_int = []
    for w in s:
        try:
            s_int.append(word2index[w.lower()])
        except KeyError:
            s_int.append(word2index['-OOV-'])

    train_sentences_X.append(s_int)

for s in test_sentences:
    s_int = []
    for w in s:
        try:
            s_int.append(word2index[w.lower()])
        except KeyError:
            s_int.append(word2index['-OOV-'])

    test_sentences_X.append(s_int)

for s in train_tags:
    train_tags_y.append([tag2index[t] for t in s])

for s in test_tags:
    test_tags_y.append([tag2index[t] for t in s])

print(train_sentences_X[0])
print(test_sentences_X[0])
print(train_tags_y[0])
print(test_tags_y[0])

MAX_LENGTH = len(max(train_sentences_X, key=len))
print(MAX_LENGTH)  # 271


train_sentences_X = pad_sequences(
    train_sentences_X, maxlen=MAX_LENGTH, padding='post')
test_sentences_X = pad_sequences(
    test_sentences_X, maxlen=MAX_LENGTH, padding='post')
train_tags_y = pad_sequences(train_tags_y, maxlen=MAX_LENGTH, padding='post')
test_tags_y = pad_sequences(test_tags_y, maxlen=MAX_LENGTH, padding='post')

print(train_sentences_X[0])
print(test_sentences_X[0])
print(train_tags_y[0])
print(test_tags_y[0])


model = Sequential()
model.add(InputLayer(input_shape=(MAX_LENGTH, )))
model.add(Embedding(len(word2index), 128))
model.add(Bidirectional(LSTM(256, return_sequences=True)))
model.add(TimeDistributed(Dense(len(tag2index))))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(0.001),
              metrics=['accuracy'])

model.summary()


def to_categorical(sequences, categories):
    cat_sequences = []
    for s in sequences:
        cats = []
        for item in s:
            cats.append(np.zeros(categories))
            cats[-1][item] = 1.0
        cat_sequences.append(cats)
    return np.array(cat_sequences)


cat_train_tags_y = to_categorical(train_tags_y, len(tag2index))
print(cat_train_tags_y[0])


scores = model.evaluate(
    test_sentences_X, to_categorical(test_tags_y, len(tag2index)))

print(f"{model.metrics_names[1]}: {scores[1] * 100}")
