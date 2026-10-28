import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ---------------- PATHS ----------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.join(BASE_DIR, "test")
MODEL_PATH = os.path.join(
    BASE_DIR, "industrial_defect_model_final.pth"
)

# ---------------- DEVICE ----------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ---------------- CHECK FILES ----------------

if not os.path.isdir(TEST_DIR):
    raise FileNotFoundError(
        f"Test folder not found: {TEST_DIR}"
    )

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )

# ---------------- IMAGE TRANSFORM ----------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------------- LOAD TEST DATA ----------------

dataset = ImageFolder(
    root=TEST_DIR,
    transform=transform
)

if len(dataset) == 0:
    raise ValueError("No images found in the test folder.")

print("\nClasses found:", dataset.classes)
print("Total test images:", len(dataset))

# ---------------- LOAD MODEL ----------------

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    )
)

model = model.to(device)
model.eval()

# ---------------- MAKE PREDICTIONS ----------------

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False
)

actual_labels = []
predicted_labels = []

with torch.no_grad():
    for images, labels in loader:
        images = images.to(device)

        outputs = model(images)
        predictions = torch.argmax(outputs, dim=1)

        actual_labels.extend(labels.tolist())
        predicted_labels.extend(predictions.cpu().tolist())

# ---------------- EVALUATION METRICS ----------------

accuracy = accuracy_score(
    actual_labels,
    predicted_labels
)

precision = precision_score(
    actual_labels,
    predicted_labels,
    average="macro",
    zero_division=0
)

recall = recall_score(
    actual_labels,
    predicted_labels,
    average="macro",
    zero_division=0
)

f1 = f1_score(
    actual_labels,
    predicted_labels,
    average="macro",
    zero_division=0
)

print("\n========== MODEL EVALUATION ==========")

print(f"Accuracy : {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall   : {recall * 100:.2f}%")
print(f"F1-score : {f1 * 100:.2f}%")

print("\nDetailed Classification Report:")
print(
    classification_report(
        actual_labels,
        predicted_labels,
        labels=list(range(len(dataset.classes))),
        target_names=dataset.classes,
        zero_division=0
    )
)

# ---------------- CONFUSION MATRIX ----------------

cm = confusion_matrix(
    actual_labels,
    predicted_labels,
    labels=list(range(len(dataset.classes)))
)

print("Confusion Matrix:")
print(cm)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=dataset.classes
)

display.plot()
plt.title("Industrial Defect Detection - Confusion Matrix")
plt.tight_layout()

# Save the confusion matrix image
output_path = os.path.join(
    BASE_DIR,
    "confusion_matrix.png"
)

plt.savefig(output_path, dpi=300)
print(f"\nConfusion matrix saved to: {output_path}")

plt.show()
