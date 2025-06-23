import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

# --- Model Definition (must match your training script) ---
class ConditionalGenerator(nn.Module):
    def __init__(self, latent_dim, num_classes=10):
        super(ConditionalGenerator, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, latent_dim)
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim * 2, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, 2, 3, bias=False),
            nn.Tanh()
        )
    def forward(self, noise, labels):
        label_embedding = self.label_embedding(labels)
        x = torch.cat([noise, label_embedding], 1)
        x = x.view(x.size(0), x.size(1), 1, 1)
        return self.main(x)

# --- Streamlit UI ---
st.title("Handwritten Digit Generator")

# Load model
@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ConditionalGenerator(100, 10).to(device)
    model.load_state_dict(torch.load('models/conditional_generator.pth', map_location=device))
    model.eval()
    return model, device

model, device = load_model()

# User input
digit = st.selectbox("Select digit to generate:", list(range(10)))
num_images = st.slider("Number of images", 1, 10, 5)
seed = st.number_input("Random seed", value=42)
torch.manual_seed(seed)

if st.button("Generate"):
    with torch.no_grad():
        noise = torch.randn(num_images, 100, device=device)
        labels = torch.full((num_images,), digit, dtype=torch.long, device=device)
        images = model(noise, labels)
        images = (images + 1) / 2.0  # Denormalize to [0,1]
        images = images.cpu().numpy()

    st.write(f"Generated images for digit {digit}:")
    cols = st.columns(num_images)
    for i in range(num_images):
        img = (images[i][0] * 255).astype(np.uint8)
        pil_img = Image.fromarray(img)
        cols[i].image(pil_img, use_container_width=True)
