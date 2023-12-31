from pickle import load

import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model


# load a clean dataset
def load_dataset(filename):
    return load(open(filename, 'rb'))


# fit a tokenizer
def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer


# calculate the maximum document length
def max_length(lines):
    return max([len(s.split()) for s in lines])


# encode a list of lines
def encode_text(tokenizer, lines, length):
    # integer encode
    encoded = tokenizer.texts_to_sequences(lines)
    # pad encoded sequences
    padded = pad_sequences(encoded, maxlen=length, padding='post')
    return padded


trainLines, trainLabels = load_dataset('train.pkl')
testLines, testLabels = load_dataset('test.pkl')
# create tokenizer
tokenizer = create_tokenizer(trainLines)
# calculate max document length
length = max_length(trainLines)
print('Max document length: %d' % length)
# calculate vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary size: %d' % vocab_size)
# encode data
trainX = encode_text(tokenizer, trainLines, length)
testX = encode_text(tokenizer, testLines, length)
trainX = np.array(trainX).astype('float32')
testX = np.array(testX).astype('float32')
trainLabels = np.array(trainLabels).astype('float32')
testLabels = np.array(testLabels).astype('float32')
# load the model
model = load_model('ngram-model.keras')
# evaluate model on training dataset
_, acc = model.evaluate([trainX, trainX, trainX], trainLabels, verbose=0)
print('Train Accuracy: %.2f' % (acc * 100))
# evaluate model on test dataset dataset
_, acc = model.evaluate([testX, testX, testX], testLabels, verbose=0)
print('Test Accuracy: %.2f' % (acc * 100))
