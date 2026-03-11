import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms  # MNIST Dataset
from torch.utils.data import DataLoader

transform = transforms.ToTensor()  # Convert images from PIL/numpy to PyTorch tensors

# Define Dataset
train_dataset = datasets.MNIST(
    root="./data",      # Folder where data will be saved
    train=True,         # Use training split (60,000 images)
    download=True,      # Download if not already present
    transform=transform # Apply tensor conversion to each image
)

# Define DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

# Define Model Class
class MNISTModel(nn.Module):

    def __init__(self):
        super(MNISTModel, self).__init__()

        self.linear1 = nn.Linear(28*28, 128)
        self.linear2 = nn.Linear(128, 10)

    def forward(self, x):

        x = x.view(-1, 28*28)  # Flatten (batch, 28, 28) → (batch, 784)

        x = self.linear1(x)
        x = torch.relu(x)

        x = self.linear2(x)

        return x

# Model
model = MNISTModel()

# Loss Function
criterion = nn.CrossEntropyLoss()  # For multi-class classification (10 digits)

# Optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training Loop
for epoch in range(2):

    for images, labels in train_loader:  # Loop through all 60,000 images in 64-image batches

        # Forward
        outputs = model(images)
        # Loss
        loss = criterion(outputs, labels)
        # Backward
        optimizer.zero_grad()
        loss.backward()
        # Update
        optimizer.step()

    print(f"Epoch done: {epoch+1}")