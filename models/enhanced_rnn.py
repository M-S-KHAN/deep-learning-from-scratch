import torch
import torch.nn as nn
import torch.nn.functional as F


class EnhancedRNN(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_layers=1,
        dropout=0.0,
        bidirectional=False,
    ):
        """
        Initialize the EnhancedRNN.

        Args:
            input_size (int): Size of input features
            hidden_size (int): Size of hidden state
            output_size (int): Size of output
            num_layers (int): Number of RNN layers
            dropout (float): Dropout probability between layers
            bidirectional (bool): If True, becomes a bidirectional RNN
        """
        super(EnhancedRNN, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1

        # RNN layers
        self.rnn = nn.RNN(
            input_size,
            hidden_size,
            num_layers,
            nonlinearity="tanh",
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )

        # Output layer
        self.fc = nn.Linear(hidden_size * self.num_directions, output_size)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size * self.num_directions)

    def forward(self, x, h_0=None):
        """
        Forward pass of the RNN.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, input_size)
            h_0 (Tensor): Initial hidden state. If None, initialized to zeros.

        Returns:
            output (Tensor): Output tensor of shape (batch_size, seq_len, output_size)
            h_n (Tensor): Final hidden state
        """
        batch_size, seq_len, _ = x.size()

        # Initialize hidden state if not provided
        if h_0 is None:
            h_0 = torch.zeros(
                self.num_layers * self.num_directions, batch_size, self.hidden_size
            ).to(x.device)

        # Apply RNN
        output, h_n = self.rnn(x, h_0)

        # Apply layer normalization
        output = self.layer_norm(output)

        # Apply dropout
        output = self.dropout(output)

        # Reshape output for bidirectional RNN if necessary
        if self.num_directions == 2:
            output = output.view(batch_size, seq_len, 2, self.hidden_size)
            output = torch.sum(output, dim=2)  # Sum bidirectional outputs

        # Apply final fully connected layer
        output = self.fc(output)

        return output, h_n
