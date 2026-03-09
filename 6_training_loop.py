import torch
import torch.nn as nn
import torch.optim as optim

# Training data
x = torch.tensor([1.0, 2.0, 3.0, 4.0]).unsqueeze(1)  # shape (4,1)
y = torch.tensor([2.0, 4.0, 6.0, 8.0]).unsqueeze(1)  # shape (4,1)

# Model
# in_features = 1 → one input value (x)
# out_features = 1 → one output value (y)
model = nn.Linear(in_features=1, out_features=1)

# Loss function
criterion = nn.MSELoss()  # ((y_pred - y)² ).mean()

# Optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(100):
    # Forward pass
    y_pred = model(x)

    # Compute loss
    loss = criterion(y_pred, y)

    # Backward pass
    optimizer.zero_grad()  # reset gradients
    loss.backward()  # compute gradients

    # Update weights
    optimizer.step()  # adjust w and b automatically

    # Print loss every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

# Final parameters
w, b = model.weight.item(), model.bias.item()
print(f"Learned parameters: w = {w:.3f}, b = {b:.3f}")