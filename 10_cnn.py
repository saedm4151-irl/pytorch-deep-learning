import torch
import torch.nn as nn

from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# Define Model Class
class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        # in: 1 grayscale channel | out: 8 feature maps | 3x3 kernel
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=8,
            kernel_size=3,
        )

        # in: 8 feature maps from conv1 | out: 16 feature maps | 3x3 kernel
        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=3,
        )

        # 2x2 window, stride 2 → halves the grid size
        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # 16 feature maps of 5x5 each, flattened to 400 → compressed to 64
        self.fc1 = nn.Linear(16 * 5 * 5, 64)

        # 64 neurons → 10 output classes (digits 0-9)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        # conv1 → relu → pool | (1,28,28) → (8,26,26) → (8,13,13)
        x = self.pool(torch.relu(self.conv1(x)))

        # conv2 → relu → pool | (8,13,13) → (16,11,11) → (16,5,5)
        x = self.pool(torch.relu(self.conv2(x)))

        # flatten (16,5,5) → 400 numbers for the linear layer
        x = x.view(-1, 16 * 5 * 5)

        # fully connected + relu
        x = torch.relu(self.fc1(x))

        # final output, 10 scores
        x = self.fc2(x)

        return x

# Transform
transform = transforms.ToTensor()

# Define Dataset
train_dataset = datasets.MNIST(
    root = "./data",
    train = True,
    download = True,
    transform = transform
)

# Define DataLoader
train_loader = DataLoader(
    dataset = train_dataset,
    batch_size = 64,
    shuffle = True,
)

# Model
model = CNN()

# Loss Function
criterion = nn.CrossEntropyLoss()

# Hyper-Parameters
lr = 0.01
epochs = 3

# Optimizer
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=lr
)

# Training loop
for epoch in range(epochs):

    for images, labels in train_loader:

        # Forward
        outputs = model(images)

        # Loss
        loss = criterion(outputs, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Update
        optimizer.step()

    print(f"Epoch: {epoch+1} Loss: {loss.item():.5f}")