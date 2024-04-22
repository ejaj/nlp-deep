import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt

from generate_positional_encodings import generate_positional_encodings
from positional_encoding import PositionalEncoding

# Set PyTorch print options to display decimal precision

torch.set_printoptions(precision=2, sci_mode=False)

sequences = [
    "the cat sat on the mat",
    "the dog barked at the cat",
    "the bird flew over the dog"
]


def tokenize(sequence):
    return sequence.split()


unique_words = set(word for seq in sequences for word in tokenize(seq))
stoi = {word: idx for idx, word in enumerate(unique_words)}

# Tokenize the sequences
tokenized_sequences = [tokenize(seq) for seq in sequences]
# Index the sequences based on the dictionary
indexed_sequences = [[stoi[word] for word in seq] for seq in tokenized_sequences]

# Convert indexed sequences to a PyTorch tensor
tensor_sequences = torch.tensor(indexed_sequences).long()

# Vocabulary size (number of unique words)
vocab_size = len(stoi)

# Embedding dimensions (number of elements in each embedding vector)
d_model = 4

# Create an embedding layer (look-up table, or lut)
lut = nn.Embedding(vocab_size, d_model)

# Embed the sequences using the embedding layer
embeddings = lut(tensor_sequences)
# Display the embedded sequences
print("Embedded Sequences:")
print(embeddings)

# # Maximum sequence length and other parameters
# max_length = 10
# d_model = 4  # Embedding dimensions
# n = 100  # Divisor for positional encoding
#
# # Generate positional encodings
# encodings = generate_positional_encodings(max_length, d_model, n)
#
# print("Positional Encodings:")
# print(encodings)
# seq_length = embeddings.shape[1]  # Get the length of the sequences (6 in this case)
# sliced_encodings = encodings[:seq_length]  # Select the first six rows of positional encodings
#
# # Display the sliced encodings
# print("Sliced Positional Encodings:")
# print(sliced_encodings)

d_model = 4
max_length = 10
dropout = 0.0

# create the positional encoding matrix
pe = PositionalEncoding(d_model, dropout, max_length)
state_di = pe.state_dict()
p_emb = pe(embeddings)


def visualize_pe(max_length, d_model, n):
    plt.imshow(generate_positional_encodings(max_length, d_model, n), aspect="auto", cmap='viridis')
    plt.title("Positional Encoding")
    plt.xlabel("Encoding Dimension")
    plt.ylabel("Position Index")

    # set the tick marks for the axes
    if d_model < 10:
        plt.xticks(torch.arange(0, d_model))
    if max_length < 20:
        plt.yticks(torch.arange(max_length - 1, -1, -1))

    plt.colorbar()
    plt.show()


# plot the encodings
max_length = 10
d_model = 4
n = 100

visualize_pe(max_length, d_model, n)
