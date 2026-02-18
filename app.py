import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import plotly.graph_objects as go

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="VeriLens AI | Image Integrity", layout="wide")

# -----------------------------
# Professional CSS
# -----------------------------
st.markdown("""
<style>
    .stApp { background-color: #ffffff; }
    .brand-container { text-align: center; padding: 20px 0px; }
    .main-brand { font-size: 52px; font-weight: 900; letter-spacing: -1px; color: #0f172a; margin-bottom: 0px; }
    .brand-subline { font-size: 16px; color: #64748b; margin-bottom: 30px; }
    .image-title { font-size: 24px; font-weight: 700; color: #1e293b; margin: 30px 0px 10px 0px; }
    .result-box { padding: 25px; border-radius: 8px; text-align: center; font-weight: 800; font-size: 30px; margin: 20px 0; box-shadow: 0 4px 12px rgba(0,0,0,0.05); }
    .status-text { text-align: center; color: #94a3b8; font-size: 18px; margin-top: 50px; font-style: italic; }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Session State for History
# -----------------------------
if 'history' not in st.session_state:
    st.session_state.history = []

# -----------------------------
# Model Setup (UPDATED TO MOBILENET)
# -----------------------------
@st.cache_resource
def load_model():
    # Changed from EfficientNet to MobileNetV2
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    
    try:
        # Update path to your calibrated model file
        model.load_state_dict(torch.load("deepfake_mobilenet_calibrated.pth", map_location="cpu"))
    except Exception as e:
        st.error(f"Error loading model: {e}")
    
    model.eval()
    return model

model = load_model()

# Standard ImageNet Normalization
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown("### Analysis Settings")
    # UPDATED: Default sensitivity to 0.80 for the calibrated model
    sensitivity = st.slider("Deepfake Sensitivity", 0.0, 1.0, 0.80)
    st.caption("Higher values reduce false positives on noisy real images (like passports).")
    
    st.markdown("---")
    st.markdown("### Analysis History")
    if st.button("Clear History", use_container_width=True):
        st.session_state.history = []
    
    for item in reversed(st.session_state.history):
        st.image(item['img'], use_container_width=True)
        st.markdown(f"**{item['label']}** ({item['conf']:.1f}%)")
        st.markdown("---")

# -----------------------------
# Header
# -----------------------------
st.markdown("""
    <div class="brand-container">
        <div class="main-brand">VeriLens AI</div>
        <div class="brand-subline">Neural Authentication & Media Integrity Engine</div>
    </div>
""", unsafe_allow_html=True)

# -----------------------------
# Main Analysis Logic
# -----------------------------
uploaded_file = st.file_uploader("Drop image here to verify authenticity", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    
    # Pre-process and Predict
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)

    p_real, p_fake = probs[0][0].item(), probs[0][1].item()
    
    # Decision logic based on the calibrated threshold
    label = "DEEPFAKE" if p_fake > sensitivity else "REAL"
    
    primary_color = "#b91c1c" if label == "DEEPFAKE" else "#15803d"
    bg_color = "#fef2f2" if label == "DEEPFAKE" else "#f0fdf4"
    
    # Display the confidence for the selected label
    confidence_val = p_fake * 100 if label == "DEEPFAKE" else p_real * 100

    if not st.session_state.history or st.session_state.history[-1]['img'] != image:
        st.session_state.history.append({"img": image, "label": label, "conf": confidence_val})

    # 1. Display Image
    st.markdown('<div class="image-title">Analysis Subject</div>', unsafe_allow_html=True)
    st.image(image, use_container_width=True)

    # 2. Result Box (Visible color & text)
    st.markdown(f"""
        <div class="result-box" style="background-color: {bg_color}; color: {primary_color}; border: 2px solid {primary_color};">
            RESULT: {label} <br>
            <span style="font-size: 20px; opacity: 0.9;">Confidence Score: {confidence_val:.2f}%</span>
        </div>
    """, unsafe_allow_html=True)

    # 3. Enhanced Charts
    graph_bg = "#f1f5f9" 
    text_color = "#1e293b" 

    col1, col2 = st.columns(2)

    with col1:
        fig_donut = go.Figure(data=[go.Pie(
            labels=["Real", "Deepfake"],
            values=[p_real, p_fake],
            hole=0.6,
            marker=dict(colors=["#16a34a", "#dc2626"], line=dict(color=graph_bg, width=2)),
            textfont=dict(color=text_color, size=14)
        )])
        fig_donut.update_layout(
            title=dict(text="Authenticity Breakdown", font=dict(color=text_color, size=18)),
            height=380, paper_bgcolor=graph_bg, plot_bgcolor=graph_bg,
            margin=dict(l=30, r=30, t=80, b=30),
            legend=dict(font=dict(color=text_color))
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with col2:
        fig_bar = go.Figure(data=[
            go.Bar(
                x=["Real", "Deepfake"],
                y=[p_real, p_fake],
                marker_color=["#16a34a", "#dc2626"],
                text=[f"{p_real*100:.1f}%", f"{p_fake*100:.1f}%"],
                textposition='auto',
                textfont=dict(color="#ffffff", size=14)
            )
        ])
        fig_bar.update_layout(
            title=dict(text="Probability Distribution", font=dict(color=text_color, size=18)),
            height=380, paper_bgcolor=graph_bg, plot_bgcolor=graph_bg,
            margin=dict(l=30, r=30, t=80, b=30),
            yaxis=dict(range=[0, 1], gridcolor="#cbd5e1", tickfont=dict(color=text_color)),
            xaxis=dict(tickfont=dict(color=text_color))
        )
        st.plotly_chart(fig_bar, use_container_width=True)

else:
    st.markdown('<div class="status-text">System ready. Standing by for image input...</div>', unsafe_allow_html=True)