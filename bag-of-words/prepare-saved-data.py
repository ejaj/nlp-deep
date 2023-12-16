import os
import re
import string
from os import listdir

from nltk.corpus import stopwords


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


# save list to file
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
    for filename in listdir(directory):
        if not filename.endswith(".txt"):
            continue
        path = directory + '/' + filename
        line = doc_to_line(path, vocab)
        lines.append(line)
    return lines


if __name__ == "__main__":
    vocab_filename = '../data/txt_sentoken/vocab.txt'
    vocab = load_doc(vocab_filename)
    vocab = vocab.split()
    vocab = set(vocab)
    # print(vocab)
    # prepare negative reviews
    negative_lines = process_docs('../data/txt_sentoken/neg', vocab)
    save_list(negative_lines, '../data/txt_sentoken/negative.txt')
    # prepare positive reviews
    positive_lines = process_docs('../data/txt_sentoken/pos', vocab)
    save_list(positive_lines, '../data/txt_sentoken/positive.txt')
