import torch

x = torch.tensor(2.0, requires_grad=True)
w = torch.tensor(5.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

y = w * x + b

y.backward() # Computes derivatives of y w.r.t w, x, and b

print("dy/dx =", x.grad)
print("dy/dw =", w.grad)
print("dy/db =", b.grad)