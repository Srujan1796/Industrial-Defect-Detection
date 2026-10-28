import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from tkinter import Tk, filedialog

# Select device
device = torch.device("cpu")

# Define the model architecture
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)

# Load trained model weights
model.load_state_dict(
    torch.load(
        "industrial_defect_model_final.pth",
        map_location=device,
        weights_only=True
    )
)

model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Select an image from your Mac
root = Tk()
root.withdraw()

image_path = filedialog.askopenfilename(
    title="Select a Casting Image",
    filetypes=[
        ("Image files", "*.jpg *.jpeg *.png *.bmp")
    ]
)

root.destroy()

if not image_path:
    print("No image selected.")
    raise SystemExit

# Load and preprocess the image
image = Image.open(image_path).convert("RGB")
image_tensor = transform(image).unsqueeze(0).to(device)

# Predict
with torch.no_grad():
    outputs = model(image_tensor)
    probabilities = torch.softmax(outputs, dim=1)
    confidence, predicted = torch.max(probabilities, dim=1)

# Dataset class order used in your project
class_names = ["def_front", "ok_front"]

prediction = class_names[predicted.item()]
confidence_percent = confidence.item() * 100

print("\n----- INDUSTRIAL DEFECT DETECTION -----")
print("Image:", image_path)
print("Prediction:", prediction)
print(f"Confidence: {confidence_percent:.2f}%")

if prediction == "def_front":
    print("Result: Defective casting detected.")
else:
    print("Result: Normal casting predicted.")
