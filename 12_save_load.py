import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Define the custom Dataset
class CusDataset(Dataset):
    def __init__(self):
        self.x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        self.y = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

dataset = CusDataset()

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)

# Defining Model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()

        self.linear1 = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear1(x)

model = SimpleNN()

criterion = nn.MSELoss()

optimizer = optim.SGD(model.parameters(), lr=0.02)

for epoch in range(200):

    for x, y in loader:
        outputs = model(x)

        loss = criterion(outputs, y)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


# Save Model
torch.save(
    model.state_dict(),
    "model.pth"
)

# Load Model
loaded_model = SimpleNN()

loaded_model.load_state_dict(torch.load("model.pth"))

loaded_model.eval()

# Test Model
with torch.no_grad():

    test = torch.tensor([[4.0]])

    prediction = loaded_model(test)

    print(prediction)