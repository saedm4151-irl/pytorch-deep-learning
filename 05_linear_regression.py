import torch

# Training Data
x = torch.tensor([1.,2.,3.,4.,5.])
y = torch.tensor([2.,4.,6.,8.,10.])

# Parameters
w = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

# Hyper-parameters
learning_rate = 0.03
epochs = 100

for epoch in range(epochs):

    # Forward
    y_pred = w * x + b

    # Loss
    loss = (y_pred - y).pow(2).mean()

    # Backward
    loss.backward()

    # Update weights and biases
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad

    # Reset gradients
    w.grad.zero_()
    b.grad.zero_()

    if epoch % 10 == 0: print(f"Epoch {epoch}: Loss {loss.item():.4f}")

print("Updated w =", w.item())
print("Updated b =", b.item())