import torch
from torch.utils.data import Dataset, DataLoader

# Define Custom Dataset
class SimpleDataset(Dataset):

    def __init__(self):
        self.x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        self.y = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

    def __len__(self): # Return the number of samples
        return len(self.x)

    def __getitem__(self, idx): # Returns one sample
        return self.x[idx], self.y[idx]


dataset = SimpleDataset()

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)

for x, y in loader:
    print(x, y)