import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Page settings
st.set_page_config(page_title="Deepfake Detector", page_icon="🕵️")

st.title("🕵Deepfake Image Detector")
st.write("Upload an image or use your camera to detect deepfakes.")

# Load model
device = torch.device("cpu")

@st.cache_resource
def load_model():
    model = models.efficientnet_b0(pretrained=False)
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features, 2
    )
    model.load_state_dict(
        torch.load("deepfake_model.pth", map_location=device)
    )
    model.eval()
    return model

model = load_model()

# Image transform
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Input method
option = st.radio("Choose input method:", ["Upload Image", "Use Camera"])

image = None

if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg","png","jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

else:
    camera_image = st.camera_input("Take a photo")
    if camera_image:
        image = Image.open(camera_image).convert("RGB")

if image:
    st.image(image, caption="Input Image", use_column_width=True)

    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, 1)

    label = "REAL" if pred.item() == 0 else "DEEPFAKE"

    st.subheader(f"Prediction: {label}")
    st.write(f"Confidence: {confidence.item()*100:.2f}%")
