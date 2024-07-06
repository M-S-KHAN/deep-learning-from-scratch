import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
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