import os
import re
import string
from collections import Counter
from os import listdir

import numpy as np
from matplotlib import pyplot
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import plot_model
from pandas import DataFrame


def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text


# turn a doc into clean tokens
def clean_doc(doc):
    # split into tokens by white space
    tokens = doc.split()
    # prepare regex for char filtering
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    # remove punctuation from each word
    tokens = [re_punc.sub('', w) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    # filter out short tokens
    tokens = [word for word in tokens if len(word) > 1]
    return tokens


# load doc and add to vocab
def add_doc_to_vocab(filename, vocab):
    # load doc
    doc = load_doc(filename)
    # clean doc
    tokens = clean_doc(doc)
    # update counts
    vocab.update(tokens)


# load doc, clean and return line of tokens
def doc_to_line(filename, vocab):
    # load the doc
    doc = load_doc(filename)
    # clean doc
    tokens = clean_doc(doc)
    # filter by vocab
    tokens = [w for w in tokens if w in vocab]
    return ' '.join(tokens)


# load all docs in a directory
def process_docs(directory, vocab):
    lines = list()
    # walk through all files in the folder
    for filename in listdir(directory):
        # skip any reviews in the test set
        if filename.startswith('cv9'):
            continue
        # create the full path of the file to open
        path = directory + '/' + filename
        # load and clean the doc
        line = doc_to_line(path, vocab)
        # add to list
        lines.append(line)
    return lines


# save list to file
def save_list(lines, filename):
    # convert lines to a single blob of text
    data = '\n'.join(lines)
    # open file
    file = open(filename, 'w')
    # write text
    file.write(data)
    # close file
    file.close()


# load and clean a dataset
def load_clean_dataset(vocab):
    # load documents
    neg = process_docs('../data/txt_sentoken/neg', vocab)
    pos = process_docs('../data/txt_sentoken/pos', vocab)

    docs = neg + pos
    # # prepare labels
    labels = [0 for _ in range(len(neg))] + [1 for _ in range(len(pos))]
    return docs, labels


# fit a tokenizer
def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer


# load all docs in a directory
def process_docs_train(directory, vocab, is_train):
    lines = list()
    # walk through all files in the folder
    for filename in listdir(directory):
        # skip any reviews in the test set
        if is_train and filename.startswith('cv9'):
            continue
        if not is_train and not filename.startswith('cv9'):
            continue
        # create the full path of the file to open
        path = directory + '/' + filename
        # load and clean the doc
        line = doc_to_line(path, vocab)
        # add to list
        lines.append(line)
    return lines


# load and clean a dataset
def load_clean_dataset_train(vocab, is_train):
    # load documents
    neg = process_docs_train('../data/txt_sentoken/neg', vocab, is_train)
    pos = process_docs_train('../data/txt_sentoken/pos', vocab, is_train)
    docs = neg + pos
    # prepare labels
    labels = [0 for _ in range(len(neg))] + [1 for _ in range(len(pos))]
    return docs, labels


def define_model(n_words):
    # define network
    model = Sequential()
    model.add(Dense(50, input_shape=(n_words,), activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # compile network
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # summarize defined model
    model.summary()
    # plot_model(model, to_file='model.png', show_shapes=True)
    return model


def prepare_data(train_docs, test_docs, mode):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_docs)
    Xtrain = tokenizer.texts_to_matrix(train_docs, mode=mode)
    Xtest = tokenizer.texts_to_matrix(test_docs, mode=mode)
    return Xtrain, Xtest


def evaluate_mode(Xtrain, ytrain, Xtest, ytest):
    scores = list()
    n_repeats = 30
    n_words = Xtest.shape[1]
    for i in range(n_repeats):
        # define network
        model = define_model(n_words)
        # fit network
        model.fit(Xtrain, ytrain, epochs=10, verbose=0)
        # evaluate
        _, acc = model.evaluate(Xtest, ytest, verbose=0)
        scores.append(acc)
        print('%d accuracy: %s' % ((i + 1), acc))
    return scores


# classify a review as negative or positive
def predict_sentiment(review, vocab, tokenizer, model):
    # clean
    tokens = clean_doc(review)
    # filter by vocab
    tokens = [w for w in tokens if w in vocab]
    # convert to line
    line = ' '.join(tokens)
    # encode
    encoded = tokenizer.texts_to_matrix([line], mode='binary')
    # predict sentiment
    yhat = model.predict(encoded, verbose=0)
    # retrieve predicted percentage and label
    percent_pos = yhat[0, 0]
    if round(percent_pos) == 0:
        return (1 - percent_pos), 'NEGATIVE'
    return percent_pos, 'POSITIVE'


if __name__ == "__main__":
    # define vocab
    # vocab = Counter()
    # add all docs to vocab
    # process_docs('../data/txt_sentoken/pos', vocab)
    # process_docs('../data/txt_sentoken/neg', vocab)
    # print the size of the vocab
    # print(len(vocab))
    # print the top words in the vocab
    # print(vocab.most_common(50))

    # keep tokens with a min occurrence
    # min_occurane = 2
    # tokens = [k for k, c in vocab.items() if c >= min_occurane]
    # print(len(tokens))
    # save tokens to a vocabulary file
    # save_list(tokens, '../data/txt_sentoken/vocab1.txt')

    # # load the vocabulary
    # vocab_filename = '../data/txt_sentoken/vocab1.txt'
    # vocab = load_doc(vocab_filename)
    # vocab = set(vocab.split())
    # # print(vocab)
    # # load all training reviews
    # docs, labels = load_clean_dataset(vocab)
    # # # summarize what we have
    # print(len(docs), len(labels))

    # load the vocabulary
    vocab_filename = '../data/txt_sentoken/vocab1.txt'
    vocab = load_doc(vocab_filename)
    vocab = set(vocab.split())
    # load all reviews
    train_docs, ytrain = load_clean_dataset_train(vocab, True)
    test_docs, ytest = load_clean_dataset_train(vocab, False)
    ytrain = np.array(ytrain).astype('float32')
    ytest = np.array(ytest).astype('float32')
    # create the tokenizer
    tokenizer = create_tokenizer(train_docs)
    # encode data
    # Xtrain = tokenizer.texts_to_matrix(train_docs, mode='freq')
    # Xtest = tokenizer.texts_to_matrix(test_docs, mode='freq')
    # n_words = Xtest.shape[1]
    # model = define_model(n_words)
    # # fit network
    # model.fit(Xtrain, ytrain, epochs=10, verbose=2)
    # # evaluate
    # loss, acc = model.evaluate(Xtest, ytest, verbose=0)
    # print('Test Accuracy: %f' % (acc * 100))

    # modes = ['binary', 'count', 'tfidf', 'freq']
    # results = DataFrame()
    # for mode in modes:
    #     # prepare data for mode
    #     Xtrain, Xtest = prepare_data(train_docs, test_docs, mode)
    #     # evaluate model on data for mode
    #     results[mode] = evaluate_mode(Xtrain, ytrain, Xtest, ytest)
    #
    # # summarize results
    # print(results.describe())
    # # plot results
    # results.boxplot()
    # pyplot.show()

    # encode data
    Xtrain = tokenizer.texts_to_matrix(train_docs, mode='binary')
    Xtest = tokenizer.texts_to_matrix(test_docs, mode='binary')
    # define network
    n_words = Xtrain.shape[1]
    model = define_model(n_words)
    # fit network
    model.fit(Xtrain, ytrain, epochs=10, verbose=2)
    # test positive text
    text = 'Best movie ever! It was great, I recommend it.'
    percent, sentiment = predict_sentiment(text, vocab, tokenizer, model)
    print('Review: [%s]\nSentiment: %s (%.3f%%)' % (text, sentiment, percent * 100))
    # test negative text
    text = 'This is a bad movie.'
    percent, sentiment = predict_sentiment(text, vocab, tokenizer, model)
    print('Review: [%s]\nSentiment: %s (%.3f%%)' % (text, sentiment, percent * 100))
