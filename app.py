import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from PIL import Image
import os

# ---------------- CONFIG ---------------- #
MODEL_PATH = "convnext_agriculture_model.pt"
CLASS_NAMES = [
    'Cherry', 'Coffee-plant', 'Cucumber', 'Fox_nut(Makhana)', 'Lemon',
    'Olive-tree', 'Pearl_millet(bajra)', 'Tobacco-plant', 'almond', 'banana',
    'cardamom', 'chilli', 'clove', 'coconut', 'cotton', 'gram',
    'jowar', 'jute', 'maize', 'mustard-oil', 'papaya', 'pineapple',
    'rice', 'soyabean', 'sugarcane', 'sunflower', 'tea', 'tomato',
    'vigna-radiati(Mung)', 'wheat'
]

NUM_CLASSES = len(CLASS_NAMES)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- MODEL LOAD ---------------- #
@st.cache_resource
def load_model():
    weights = ConvNeXt_Tiny_Weights.DEFAULT
    model = convnext_tiny(weights=weights)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# ---------------- IMAGE TRANSFORM ---------------- #
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------------- STREAMLIT UI ---------------- #
st.title("🌾 ConvNeXt Agricultural Crop Classifier")
st.write("Upload an image of a crop to classify it using a trained ConvNeXt-Tiny model.")

uploaded_file = st.file_uploader("Upload an image (JPG/PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📷 Uploaded Image", use_column_width=True)

    # Process and predict
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)
        class_name = CLASS_NAMES[predicted.item()]

    st.success(f"✅ **Predicted Class:** {class_name}")
    st.info(f"📈 **Confidence:** {confidence.item() * 100:.2f}%")

