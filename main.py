from models.rnn import SimpleRNN

import torch
import torch.nn as nn


if __name__ == "__main__":

    # Define hyperparameters
    input_size = 10
    hidden_size = 20
    output_size = 5
    sequence_length = 15
    batch_size = 32

    # Instantiate the model
    model = SimpleRNN(input_size, hidden_size, output_size)

    # Create Sample data
    x = torch.randn(batch_size, sequence_length, input_size)
    y = torch.randn(batch_size, sequence_length, output_size)

    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Number of epochs
    num_epochs = 100

    for epoch in range(num_epochs):

        # Forward pass
        outputs, _ = model(x)
        loss = criterion(outputs, y)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
