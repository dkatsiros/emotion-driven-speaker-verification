import torch
from torch import nn
import numpy as np


class LSTM(nn.Module):
    """LSTM implementation."""

    def __init__(self, input_size, hidden_size=16, output_size=7, num_layers=1,
                 bidirectional=False, dropout=0):

        super(LSTM, self).__init__()

        # No embedding layer
        self.input_size = input_size
        self.bidirectional = bidirectional

        # Define the LSTM layers
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True,
                            dropout=dropout,
                            bidirectional=self.bidirectional)

        # Final layer
        self.feature_size = (self.hidden_size *
                             2 if self.bidirectional else self.hidden_size)
        self.linear = nn.Linear(self.feature_size, output_size)
        self.softmax = nn.Softmax()

    def forward(self, x, lengths):
        """
        This is the heart of the model.
        This function, defines how the data passes through the network.
        Arguments:
            x : 3D numpy array of dimension N x L x D
                N: batch index
                L: sequence index
                D: feature index

        Returns: the logits for each class
        """
        # Obtain the model's device ID
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Get size of the given batch
        batch_size = len(x)
        # Set initial hidden and cell states for the lstm layer.
        h0 = torch.zeros(self.num_layers * 2 if self.bidirectional else self.num_layers,
                         x.size(0),
                         self.hidden_size).to(DEVICE)
        c0 = torch.zeros(self.num_layers * 2 if self.bidirectional else self.num_layers,
                         x.size(0),
                         self.hidden_size).to(DEVICE)

        # Pass data through LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Remove sequence dimension, keep only last idx
        last = self.last_timestep(
            out, lengths.to(DEVICE), self.bidirectional)
        # last = self.last_by_index(
        #     out, torch.tensor(lengths).to(DEVICE))  # (N,D)

        # maybe by concatenation of the last,mean_pool,max_pool
        # representations = torch.cat((last, mean_pool, max_pool), 1)

        # Pass the output of LSTM to the remaining network, projecting to classes
        representations = self.linear(last)
        return representations
        # Convert to probabilities
        logits = self.softmax(representations)

        return logits

    def last_timestep(self, outputs, lengths, bidirectional=False):
        """
            Returns the last output of the LSTM taking into account the zero padding
        """
        if bidirectional:
            forward, backward = self.split_directions(outputs)
            last_forward = self.last_by_index(forward, lengths)
            last_backward = backward[:, 0, :]
            # Concatenate and return - maybe add more functionalities like average
            return torch.cat((last_forward, last_backward), dim=-1)
        else:
            return self.last_by_index(outputs, lengths)

    @ staticmethod
    def last_by_index(outputs, lengths):
        # Index of the last output for each sequence.
        idx = (lengths - 1).view(-1, 1).expand(outputs.size(0),
                                               outputs.size(2)).unsqueeze(1)
        return outputs.gather(1, idx).squeeze()

    @ staticmethod
    def split_directions(outputs):
        direction_size = int(outputs.size(-1) / 2)
        forward = outputs[:, :, :direction_size]
        backward = outputs[:, :, direction_size:]
        return forward, backward

    @ staticmethod
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
