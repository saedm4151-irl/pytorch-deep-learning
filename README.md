# pytorch-deep-learning

Continuing from building a neural network from scratch using only NumPy,
this repo documents my hands-on journey learning PyTorch — from tensors
all the way to training real models on real datasets.

Previous repo: [neural-network-from-scratch](https://github.com/muhammedsaed/neural-network-from-scratch)

---

## File Structure

| # | File | Description |
|---|------|-------------|
| 1 | `1_tensor_init_and_operations.py` | Tensor creation, operations, basic vector/matrix math |
| 2 | `2_backward.py` | Single & multiple variable backward pass, requires_grad |
| 3 | `3_autograd_graph.py` | Computation graph, grad_fn, chain rule visualization |
| 4 | `4_computational_graph.png` | Hand-drawn computation graph diagram |
| 5 | `5_linear_regression.py` | Manual linear regression — forward, loss, backward, gradient update |
| 6 | `6_training_loop.py` | Real training loop with nn.Linear, optimizer, loss, zero_grad |
| 7 | `7_nn_module.py` | Neural network class using nn.Module and forward() |
| 8 | `8_dataloader.py` | Dataset and DataLoader — batching and shuffling |
| 9 | `9_mnist_model.py` | First real NN on MNIST — classification, accuracy, softmax |
| 10 | `10_cnn.py` | Convolutional neural network for image classification |
| 11 | `11_cnn.png` | Hand-drawn forward pass visualization of CNN Architecture |
| 12 | `12_save_load.py` | Save and load models with torch.save / torch.load |

---

## Progress

- [x] Tensors and operations
- [x] Backward pass and requires_grad
- [x] Autograd and computation graph
- [x] Manual linear regression
- [x] Real training loop with nn.Linear
- [ ] Neural network with nn.Module ← currently here
- [ ] Dataset and DataLoader
- [ ] MNIST classification
- [ ] Convolutional neural network
- [ ] Save and load models

---

## Roadmap Beyond This Repo

- [ ] CBIR deep learning model for medical image retrieval (research project)
- [ ] Computer vision pipeline for AI-powered Lost & Found system
- [ ] Mini-batch SGD and advanced optimizers

---

## Why PyTorch

After implementing backpropagation and vectorization completely by hand
in the previous repo, PyTorch is the natural next step for my AI coursework
and ongoing research on CBIR-based medical image retrieval with my professor.

---

## Stack

- Python 3
- PyTorch
- NumPy
- PyCharm (virtual environment)

---

## Author

**Muhammed Saed**
AI Student — American University of Ras Al Khaimah, UAE
muhammedsaed.uni@gmail.com
