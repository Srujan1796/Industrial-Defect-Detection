import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

st.set_page_config(
    page_title="Industrial Defect Detection",
    page_icon="🔍"
)

st.title("Industrial Defect Detection")
st.write("Upload an industrial product image to check for defects.")

@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    model.load_state_dict(
        torch.load(
            "industrial_defect_model_final.pth",
            map_location="cpu",
            weights_only=True
        )
    )

    model.eval()
    return model

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", width=400)

    if st.button("Check Image"):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])

        model = load_model()
        image_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, prediction = probabilities.max(dim=1)

        classes = ["def_front", "ok_front"]
        result = classes[prediction.item()]

        st.subheader("Prediction Result")
        st.write("Class:", result)
        st.write(f"Confidence: {confidence.item() * 100:.2f}%")
