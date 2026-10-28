# Industrial Defect Detection System

## Overview
This project uses deep learning to classify industrial casting images as defective or normal. It uses a ResNet18 convolutional neural network trained for binary image classification.

## Features
- Upload and preview casting images.
- Predict whether a casting is defective or normal.
- Display prediction confidence.
- Save inspection results to a CSV history file.
- View previous inspection records.
- Generate PDF inspection reports.
- Compare the original image with the prediction result.
- Evaluate the model using accuracy, precision, recall, F1-score, and a confusion matrix.

## Technologies Used
- Python
- PyTorch
- Torchvision
- Tkinter
- Pillow
- Matplotlib
- Scikit-learn
- ReportLab

## Model Evaluation
The model was evaluated on 715 test images.

| Metric | Result |
|---|---:|
| Accuracy | 99.02% |
| Precision (weighted) | 99% approximately |
| Recall (weighted) | 99% approximately |
| F1-score (weighted) | 99% approximately |

The confusion matrix recorded 708 correct predictions and 7 incorrect predictions.

These results apply to the evaluated test dataset and do not guarantee the same performance on new industrial images.

## Project Files
- `app.py` — desktop application and inspection interface.
- `predict.py` — prediction script.
- `evaluate_model.py` — model evaluation script.
- `industrial_defect_model_final.pth` — trained model weights.

## Installation

### 1. Clone the repository
```bash
git clone YOUR_REPOSITORY_URL
cd IndustrialDefectDetection
```

Replace `YOUR_REPOSITORY_URL` with your GitHub repository URL.

### 2. Create a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install the required libraries
```bash
pip install torch torchvision pillow reportlab scikit-learn matplotlib
```

### 4. Run the application
```bash
python3 app.py
```

## Usage
1. Open the application.
2. Click **UPLOAD IMAGE**.
3. Select a casting image.
4. Click **PREDICT DEFECT**.
5. Use **COMPARE IMAGE & RESULT** to view the image and prediction side by side.
6. Use **VIEW HISTORY** to see saved inspections.
7. Use **GENERATE PDF REPORT** to save an inspection report.

## Disclaimer
This is an educational project. AI predictions should be independently verified before being used for real industrial quality-control decisions.