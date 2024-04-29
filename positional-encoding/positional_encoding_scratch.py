import numpy as np

# Parameters
d = 16  # Embedding dimension
max_len = 10  # Maximum sequence length

# Generate positional encoding
pos_enc = np.zeros((max_len, d))  # Initialize a zero matrix for positional encoding

# print(pos_enc)

for pos in range(max_len):
    for i in range(0, d, 2):  # Process even indices
        div_val = 10000 * pos + i
        div_term = 10000 ** (2 * i / d)  # Division term for scaling
        pos_enc[pos, i] = np.sin(pos / div_term)  # Sine for even indices
        if i + 1 < d:  # Ensure we don't go beyond the last dimension
            pos_enc[pos, i + 1] = np.cos(pos / div_term)  # Cosine for odd indices

# Display the positional encoding
print("Positional Encoding Matrix:")
print(pos_enc)
