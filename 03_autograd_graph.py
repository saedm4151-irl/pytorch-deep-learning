import torch

x = torch.tensor(2.0, requires_grad=True)
w = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

y = w * x + b

print(y)

y.backward() # Automatically Computes all the necessary derivatives

print("dy/dx =", x.grad)
print("dy/dw =", w.grad)
print("dy/db =", b.grad)

# Last operation that CREATED y was (+), so grad_fn = AddBackward
print(y.grad_fn)

# Last operation that CREATED it was (*), so grad_fn = MulBackward
print((w * x).grad_fn)