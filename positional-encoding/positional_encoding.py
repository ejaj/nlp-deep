import math

import torch
from torch import nn, Tensor


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_length: int = 5000):
        """
        :param d_model: dimension of embedding vectors
        :param dropout: randomly zeroes-out some of the input
        :param max_length: maximum length of input sequence
        """
        super(PositionalEncoding, self).__init__()
        # initialize dropout
        self.dropout = nn.Dropout(p=dropout)
        # create tensor of 0s
        pe = torch.zeros(max_length, d_model)

        # create position column
        k = torch.arange(0, max_length).unsqueeze(1)

        # calc divisor for positional encoding
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )

        # calc sine on even indices
        pe[:, 0::2] = torch.sin(k * div_term)

        # calc cosine on odd indices
        pe[:, 1::2] = torch.cos(k * div_term)

        # add dimension
        pe = pe.unsqueeze(0)

        # buffers are saved in state_dict but not trained by the optimizer
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor):
        """
        :param x:  embeddings (batch_size, seq_length, d_model)
        :return: embeddings + positional encodings (batch_size, seq_length, d_model)
        """
        # add positional encoding to the embeddings
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)

# d_model = 4
# max_length = 10
# dropout = 0.0
#
# # create the positional encoding matrix
# pe = PositionalEncoding(d_model, dropout, max_length)
#
# # preview the values
# state_di = pe.state_dict()
# print(state_di)
