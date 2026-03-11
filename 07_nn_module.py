import torch
import torch.nn as nn
import torch.optim as optim


# Define Model
class SimpleNN(nn.Module):

    def __init__(self):
        super(SimpleNN, self).__init__()

        self.linear1 = nn.Linear(1, 4)  # Hidden Layer
        self.linear2 = nn.Linear(4, 1)  # Output Layer

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x


# Training Data
x = torch.tensor([1., 2., 3., 4.]).unsqueeze(1)
y = torch.tensor([2., 4., 6., 8.]).unsqueeze(1)

# Model
model = SimpleNN()

# Print all parameters
for name, param in model.named_parameters():
    print(name, param)

# Loss
criterion = nn.MSELoss()

# Optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training Loop
for epoch in range(100):
    # Forward
    y_pred = model(x)

    # Loss
    loss = criterion(y_pred, y)

    # Backward
    optimizer.zero_grad()
    loss.backward()

    # Update
    optimizer.step()

    if (epoch + 1) % 10 == 0: print(f"epoch: {epoch+1} loss: {loss.item():.5f}")