import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import gradio as gr

device = torch.device("cpu")

# Load model
model = models.efficientnet_b0(pretrained=False)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)

model.load_state_dict(
    torch.load("deepfake_model.pth", map_location=device)
)
model.eval()

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def predict(image):
    image = image.convert("RGB")
    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        out = model(img)
        prob = torch.softmax(out, dim=1)
        conf, pred = torch.max(prob, 1)

    label = "REAL" if pred.item() == 0 else "DEEPFAKE"
    return label, round(conf.item()*100, 2)

interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", source="webcam"),
    outputs=[
        gr.Label(label="Prediction"),
        gr.Number(label="Confidence (%)")
    ],
    title="Deepfake Image Detector",
    description="Upload an image or take a photo to detect deepfakes"
)

interface.launch()

