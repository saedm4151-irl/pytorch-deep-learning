import torch

# From Python List
x = torch.tensor([1,2,3])

# With Zeroes
x = torch.zeros(3)

# With Ones
x = torch.ones(5)

# With Random values (0-1)
x = torch.rand(3)

# From Matrix
x = torch.tensor([
	[1,2],
	[2,3]
	])

a = torch.tensor([1.0,2.0])
b = torch.tensor([3.0,4.0])

# Addition
print(a + b)

# Multiplication
print(a * b)

# Division
print(a / b)

# Subtraction
print(a - b)