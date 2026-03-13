# pytorch-deep-learning

Continuing from building a neural network from scratch using only NumPy,
this repo documents my hands-on journey learning PyTorch — from tensors
all the way to training and deploying real models on real datasets.

Previous repo: [neural-network-from-scratch](https://github.com/muhammedsaed/neural-network-from-scratch)

Live demo: [MNIST Digit Recognizer](https://pytorch-mnist-model.streamlit.app)

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
| 11 | `11_cnn_architecture.png` | Hand-drawn forward pass visualization of CNN architecture |
| 12 | `12_save_load.py` | Save and load models with torch.save / torch.load |
| 13 | `13_full_model.py` | Complete model — training, validation, accuracy, save and load |
| 14 | `14_full_model_architecture.png` | Architecture diagram of the full model |
| — | `README.md` | Repository overview, file index, and learning roadmap |
| — | `app.py` | Streamlit web app — drawable canvas, inference, confidence display |
| — | `mnist_cnn.pth` | Saved weights for the MNIST CNN (~98% validation accuracy) |
| — | `model.pth` | Saved weights for the simple regression model |
| — | `requirements.txt` | Dependencies for Streamlit Cloud deployment |

---

## Live Demo

A drawable digit recognizer built on top of the trained CNN.
Draw any digit 0–9 on the canvas and the model returns a prediction with confidence scores for all 10 classes.

Deployed on Streamlit Cloud: [pytorch-mnist-model.streamlit.app](https://pytorch-mnist-model.streamlit.app)

Note: The model achieves ~98% accuracy on the MNIST test set but shows reduced accuracy on hand-drawn input.
This is a known distribution gap — the model was trained on clean, centered MNIST digits and has not seen
real handwriting styles. The fix is data augmentation during training, which is a planned next step.

---

## Progress

- [x] Tensors and operations
- [x] Backward pass and requires_grad
- [x] Autograd and computation graph
- [x] Manual linear regression
- [x] Real training loop with nn.Linear
- [x] Neural network with nn.Module
- [x] Dataset and DataLoader
- [x] MNIST classification
- [x] Convolutional neural network
- [x] Save and load models
- [x] Full training + validation loop with accuracy tracking
- [x] Model deployment — live Streamlit app

---

## Roadmap Beyond This Repo

- [ ] Data augmentation to improve real-world generalization
- [ ] CBIR deep learning model for medical image retrieval (research project)

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
- Streamlit
- PIL (Pillow)
- PyCharm (virtual environment)

---

## Author

**Muhammed Saed**
AI Student — American University of Ras Al Khaimah, UAE
muhammedsaed.uni@gmail.com
