import numpy as np
import math
import torch


# Function to generate positional encodings
def generate_positional_encodings(max_length, d_model, n):
    # calculate the div_term
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(n) / d_model))

    # generate the positions into a column matrix
    k = torch.arange(0, max_length).unsqueeze(1)

    # generate an empty tensor
    pe = torch.zeros(max_length, d_model)

    # set the even values
    pe[:, 0::2] = torch.sin(k * div_term)

    # set the odd values
    pe[:, 1::2] = torch.cos(k * div_term)

    # add a dimension
    pe = pe.unsqueeze(0)

    return pe


# Maximum sequence length and other parameters
max_length = 10
d_model = 4  # Embedding dimensions
n = 100  # Divisor for positional encoding

# Generate positional encodings
encodings = generate_positional_encodings(max_length, d_model, n)

print("Positional Encodings:")
print(encodings)
