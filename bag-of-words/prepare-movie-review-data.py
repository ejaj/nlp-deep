import os
import re
import string
from collections import Counter
from os import listdir
from nltk.corpus import stopwords


def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text


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


def add_doc_to_voc(filename, vocab):
    doc = load_doc(filename)
    tokens = clean_doc(doc)
    vocab.update(tokens)


def process_docs(dirs, vocab):
    for filename in listdir(dirs):
        if not filename.endswith(".txt"):
            continue
        path = dirs + '/' + filename
        add_doc_to_voc(path, vocab)


def save_list(lines, filename):
    # Ensure the directory exists
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Join the lines with newline characters
    data = '\n'.join(lines)

    # Open the file in write mode
    with open(filename, 'w') as file:
        file.write(data)


if __name__ == "__main__":
    # filename = '../data/txt_sentoken/neg/cv917_29484.txt'
    # text = load_doc(filename)

    # directory = '../data/txt_sentoken/neg'
    # process_docs(directory)
    # add all docs to vocab
    # define vocab
    vocab = Counter()
    process_docs('../data/txt_sentoken/neg', vocab)
    process_docs('../data/txt_sentoken/pos', vocab)
    # print the size of the vocab
    print(len(vocab))
    print(vocab.most_common(50))

    min_occurrence = 5
    tokens = [k for k, c in vocab.items() if c >= min_occurrence]
    save_list(tokens, '../data/txt_sentoken/vocab.txt')
