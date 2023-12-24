from pickle import load

import numpy as np
from keras.models import load_model
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences


def generate_seq(model, mapping, seq_length, seed_text, n_chars):
    in_text = seed_text
    # generate a fixed number of characters
    for _ in range(n_chars):
        # encode the characters as integers
        encoded = [mapping[char] for char in in_text]
        # truncate sequences to a fixed length
        encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
        # one hot encode
        encoded = to_categorical(encoded, num_classes=len(mapping))
        # reshape to be compatible with the model input
        encoded = encoded.reshape(1, seq_length, -1)
        # predict character
        yhat = model.predict(encoded, verbose=0)
        # get the character with the highest probability
        yhat = np.argmax(yhat, axis=-1)
        out_char = ''
        for char, index in mapping.items():
            if index == yhat:
                out_char = char
                break
        # append to input
        in_text += out_char
    return in_text


# load the model
model = load_model('model.h5')
# load the mapping
mapping = load(open('mapping.pkl', 'rb'))

# Test the function
print(generate_seq(model, mapping, 10, 'Sing a son', 20))
print(generate_seq(model, mapping, 10, 'king was i', 20))
print(generate_seq(model, mapping, 10, 'hello worl', 20))
