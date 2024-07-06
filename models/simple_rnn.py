import torch
import torch.nn as nn


class SimpleRNN(nn.Module):
    """
    A simple Recurrent Neural Network (RNN) implementation.

    This RNN processes sequences of input data, maintaining a hidden state
    that is updated at each time step. It produces an output at each time step.

    Attributes:
        hidden_size (int): The number of features in the hidden state.
        W_ih (nn.Linear): Weight matrix for input-to-hidden connections.
        W_hh (nn.Linear): Weight matrix for hidden-to-hidden connections.
        b_h (nn.Parameter): Bias for the hidden state.
        W_ho (nn.Linear): Weight matrix for hidden-to-output connections.
        tanh (nn.Tanh): Hyperbolic tangent activation function.
    """

    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize the SimpleRNN.

        Args:
            input_size (int): The number of expected features in the input x.
            hidden_size (int): The number of features in the hidden state h.
            output_size (int): The number of expected features in the output.
        """
        super(SimpleRNN, self).__init__()

        self.hidden_size = hidden_size

        # Input to Hidden Weights
        self.W_ih = nn.Linear(input_size, hidden_size, bias=False)

        # Hidden to Hidden Weights
        self.W_hh = nn.Linear(hidden_size, hidden_size, bias=False)

        # Hidden bias
        self.b_h = nn.Parameter(torch.zeros(hidden_size))

        # Hidden to Output weights
        self.W_ho = nn.Linear(hidden_size, output_size)

        # Activation function
        self.tanh = nn.Tanh()

    def forward(self, x):
        """
        Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, sequence_length, input_size).

        Returns:
            tuple: A tuple containing:
                - outputs (torch.Tensor): The output tensor of shape (batch_size, sequence_length, output_size).
                - h (torch.Tensor): The final hidden state of shape (batch_size, hidden_size).
        """

        # x shape: (batch_size, sequence_length, input_size)
        batch_size, sequence_length, _ = x.size()

        # initialize the hidden state
        h = torch.zeros(batch_size, self.hidden_size).to(x.device)

        # List to store outputs
        outputs = []

        for t in range(sequence_length):

            # Get the input at this time step
            x_t = x[:, t, :]

            # Compute the hidden state
            h = self.tanh(self.W_ih(x_t) + self.W_hh(h) + self.b_h)

            # Compute the output
            out = self.W_ho(h)

            outputs.append(out)

        # Stack outputs to a tensor
        outputs = torch.stack(outputs, dim=1)

        return outputs, h
