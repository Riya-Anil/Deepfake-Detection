import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Deepfake Image Detector",
    page_icon="🕵️",
    layout="centered"
)

st.title("🕵️ Deepfake Image Detector")
st.write("Upload an image to check whether it is REAL or a DEEPFAKE.")

# -----------------------------
# Device (CPU)
# -----------------------------
device = torch.device("cpu")

# -----------------------------
# Load model (EfficientNet-B0)
# -----------------------------
@st.cache_resource
def load_model():
    model = models.efficientnet_b0(pretrained=False)
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features, 2
    )
    model.load_state_dict(
        torch.load("deepfake_efficientnet_finetuned.pth", map_location=device)
    )
    model.eval()
    return model

model = load_model()
st.success("Model loaded successfully")

# -----------------------------
# Transform (MATCH TRAINING)
# -----------------------------
transform = transforms.Compose([
    transforms.Resize(160),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# -----------------------------
# Upload image
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "png", "jpeg"]
)

# -----------------------------
# Prediction
# -----------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)

    p_real = probs[0][0].item()
    p_fake = probs[0][1].item()

    # -----------------------------
    # Decision logic
    # -----------------------------
    if max(p_real, p_fake) < 0.60:
        label = "UNCERTAIN"
    elif p_fake > p_real:
        label = "DEEPFAKE"
    else:
        label = "REAL"

    st.markdown("---")
    st.subheader(f"🔍 Prediction: **{label}**")
    st.write(f"Real: {p_real*100:.2f}% | Fake: {p_fake*100:.2f}%")

    if label == "UNCERTAIN":
        st.warning(
            "⚠️ Low confidence. The image lies near the decision boundary."
        )

    st.caption(
        "⚠️ Model trained on benchmark deepfake datasets. "
        "Predictions are most reliable for images similar to the training data."
    )

    # -----------------------------
    # Debug (optional)
    # -----------------------------
    with st.expander("Debug details"):
        st.write("Raw outputs:", outputs)
        st.write("Softmax probabilities:", probs)

else:
    st.info("Please upload an image to begin.")
