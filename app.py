import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from streamlit_drawable_canvas import st_canvas

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.fc1 = nn.Linear(32*5*5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32*5*5)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

@st.cache_resource
def load_model():
    model = CNN()
    model.load_state_dict(torch.load("mnist_cnn.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

st.title("MNIST Digit Recognizer")
st.write("Draw a digit below and the model will predict it.")

canvas = st_canvas(
    fill_color="black",
    stroke_width=12,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas"
)

if canvas.image_data is not None:
    img = Image.fromarray(canvas.image_data.astype("uint8"), "RGBA")
    img = img.convert("L")
    img = img.resize((28, 28))
    img_array = np.array(img) / 255.0

    if img_array.max() > 0.1:
        tensor = torch.tensor(img_array, dtype=torch.float32)
        tensor = tensor.unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            output = model(tensor)
            probs = torch.softmax(output, dim=1)[0]
            predicted = torch.argmax(probs).item()
            confidence = probs[predicted].item()

        st.markdown(f"### Predicted digit: **{predicted}**")
        st.write(f"Confidence: {confidence*100:.1f}%")

        st.write("All probabilities:")
        for i, p in enumerate(probs):
            st.progress(float(p), text=f"{i}: {p*100:.1f}%")
