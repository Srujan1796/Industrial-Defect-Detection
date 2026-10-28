import os
import csv
from io import BytesIO
from datetime import datetime

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as PDFImage
)


# ---------------- PAGE SETUP ----------------

st.set_page_config(
    page_title="Industrial Defect Detection",
    page_icon="🔍",
    layout="wide"
)


# ---------------- CUSTOM STYLING ----------------

st.markdown("""
<style>
    .stApp {
        background-color: #E8EDF3;
    }

    .main-title {
        background-color: #14213D;
        color: white;
        padding: 25px;
        text-align: center;
        border-radius: 10px;
        margin-bottom: 20px;
    }

    .main-title h1 {
        color: white;
    }

    .main-title p {
        color: #DCE7F5;
    }

    .result-card {
        background-color: white;
        padding: 25px;
        border-radius: 10px;
        border: 1px solid #C5D0DC;
        text-align: center;
    }

    .normal-result {
        color: #187A3D;
        font-size: 25px;
        font-weight: bold;
    }

    .defect-result {
        color: #B42318;
        font-size: 25px;
        font-weight: bold;
    }

    .section-title {
        color: #14213D;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# ---------------- HEADER ----------------

st.markdown("""
<div class="main-title">
    <h1>INDUSTRIAL DEFECT DETECTION</h1>
    <p>AI-POWERED CASTING INSPECTION | RESNET18</p>
</div>
""", unsafe_allow_html=True)


st.write(
    "Upload a casting image to inspect it using the trained "
    "deep learning model."
)


# ---------------- MODEL SETUP ----------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "industrial_defect_model_final.pth"
)

DEVICE = torch.device("cpu")

CLASSES = ["def_front", "ok_front"]


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Model file not found. Make sure "
            "industrial_defect_model_final.pth is in the repository."
        )

    model = models.resnet18(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        2
    )

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE,
            weights_only=True
        )
    )

    model.eval()

    return model


# ---------------- IMAGE PREPROCESSING ----------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ---------------- SESSION STATE ----------------

if "history" not in st.session_state:
    st.session_state.history = []

if "image_bytes" not in st.session_state:
    st.session_state.image_bytes = None

if "image_name" not in st.session_state:
    st.session_state.image_name = None

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "confidence" not in st.session_state:
    st.session_state.confidence = None

if "inspection_time" not in st.session_state:
    st.session_state.inspection_time = None


# ---------------- IMAGE UPLOAD ----------------

st.subheader("1. Upload Casting Image")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "bmp"],
    key="image_uploader"
)


# ---------------- HANDLE NEW IMAGE ----------------

if uploaded_file is not None:

    new_image_bytes = uploaded_file.getvalue()

    if (
        st.session_state.image_bytes != new_image_bytes
        or st.session_state.image_name != uploaded_file.name
    ):
        st.session_state.image_bytes = new_image_bytes
        st.session_state.image_name = uploaded_file.name
        st.session_state.prediction = None
        st.session_state.confidence = None
        st.session_state.inspection_time = None


# ---------------- PREDICTION BUTTON ----------------

if st.session_state.image_bytes is not None:

    try:
        original_image = Image.open(
            BytesIO(st.session_state.image_bytes)
        ).convert("RGB")

        left, right = st.columns(2)

        with left:
            st.markdown(
                "### Casting Image Preview"
            )

            st.image(
                original_image,
                caption=st.session_state.image_name,
                use_container_width=True
            )

        with right:
            st.markdown(
                "### Inspection Control"
            )

            st.write(
                "Run the model to inspect the uploaded image."
            )

            if st.button(
                "PREDICT DEFECT",
                type="primary",
                use_container_width=True
            ):

                with st.spinner(
                    "Analyzing image..."
                ):

                    try:
                        model = load_model()

                        image_tensor = transform(
                            original_image
                        ).unsqueeze(0).to(DEVICE)

                        with torch.no_grad():

                            outputs = model(image_tensor)

                            probabilities = torch.softmax(
                                outputs,
                                dim=1
                            )

                            confidence, predicted = torch.max(
                                probabilities,
                                dim=1
                            )

                        predicted_class = CLASSES[
                            predicted.item()
                        ]

                        confidence_percent = (
                            confidence.item() * 100
                        )

                        if predicted_class == "def_front":
                            prediction_text = "DEFECTIVE CASTING"
                        else:
                            prediction_text = "NORMAL CASTING"

                        inspection_time = datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        )

                        st.session_state.prediction = prediction_text

                        st.session_state.confidence = (
                            confidence_percent
                        )

                        st.session_state.inspection_time = (
                            inspection_time
                        )

                        # Save this inspection to session history.
                        st.session_state.history.append({
                            "Date and Time": inspection_time,
                            "Image Name": st.session_state.image_name,
                            "Prediction": prediction_text,
                            "Confidence": (
                                f"{confidence_percent:.2f}%"
                            )
                        })

                    except Exception as error:
                        st.error(
                            f"Prediction failed: {error}"
                        )


# ---------------- DISPLAY RESULT ----------------

if st.session_state.prediction is not None:

    st.divider()

    st.subheader("2. Inspection Result")

    result_left, result_right = st.columns(2)

    with result_left:

        st.markdown(
            "### Prediction"
        )

        if (
            st.session_state.prediction
            == "DEFECTIVE CASTING"
        ):
            st.markdown(
                '<div class="result-card">'
                '<div class="defect-result">'
                'DEFECTIVE CASTING'
                '</div></div>',
                unsafe_allow_html=True
            )

            st.error(
                "The model classified this image "
                "as a defective casting."
            )

        else:
            st.markdown(
                '<div class="result-card">'
                '<div class="normal-result">'
                'NORMAL CASTING'
                '</div></div>',
                unsafe_allow_html=True
            )

            st.success(
                "The model classified this image "
                "as a normal casting."
            )

    with result_right:

        st.markdown(
            "### Confidence"
        )

        st.metric(
            "Prediction Confidence",
            f"{st.session_state.confidence:.2f}%"
        )

        st.progress(
            min(
                max(
                    int(st.session_state.confidence),
                    0
                ),
                100
            )
        )

        st.caption(
            "Confidence is the model's softmax score, "
            "not a guarantee that the prediction is correct."
        )

    st.info(
        "Date and time: "
        + str(st.session_state.inspection_time)
    )


# ---------------- FEATURE TABS ----------------

st.divider()

tab_compare, tab_history, tab_report = st.tabs([
    "🖼️ Compare Image & Result",
    "📋 View History",
    "📄 Generate PDF Report"
])


# ---------------- COMPARE IMAGE & RESULT ----------------

with tab_compare:

    st.subheader("Casting Inspection Comparison")

    if st.session_state.prediction is None:

        st.info(
            "Upload an image and click PREDICT DEFECT "
            "to view the comparison."
        )

    else:

        original_column, result_column = st.columns(2)

        with original_column:

            st.markdown(
                "### Original Casting Image"
            )

            st.image(
                Image.open(
                    BytesIO(st.session_state.image_bytes)
                ).convert("RGB"),
                use_container_width=True
            )

            st.caption(
                st.session_state.image_name
            )

        with result_column:

            st.markdown(
                "### AI Inspection Result"
            )

            st.write(
                st.session_state.prediction
            )

            st.write(
                "Confidence: "
                f"{st.session_state.confidence:.2f}%"
            )

            if (
                st.session_state.prediction
                == "DEFECTIVE CASTING"
            ):
                st.error(
                    "Defective casting detected by the model."
                )
            else:
                st.success(
                    "Normal casting predicted by the model."
                )

        st.warning(
            "AI predictions should be verified before "
            "making important industrial decisions."
        )


# ---------------- VIEW PREDICTION HISTORY ----------------

with tab_history:

    st.subheader("Prediction History")

    if not st.session_state.history:

        st.info(
            "No inspection records yet. "
            "Upload an image and run a prediction."
        )

    else:

        st.dataframe(
            st.session_state.history,
            use_container_width=True,
            hide_index=True
        )

        csv_buffer = BytesIO()

        csv_text = csv.DictWriter

        import pandas as pd

        history_df = pd.DataFrame(
            st.session_state.history
        )

        st.download_button(
            label="Download History as CSV",
            data=history_df.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="prediction_history.csv",
            mime="text/csv"
        )

        st.caption(
            "History is maintained for this browser session. "
            "It is not a permanent shared database."
        )


# ---------------- GENERATE PDF REPORT ----------------

with tab_report:

    st.subheader("Generate Inspection PDF Report")

    if st.session_state.prediction is None:

        st.info(
            "Run a prediction first to generate a report."
        )

    else:

        st.write(
            "Create a PDF containing the inspected image, "
            "prediction, confidence score, date, and model name."
        )

        if st.button(
            "GENERATE PDF REPORT",
            type="primary"
        ):

            try:

                pdf_buffer = BytesIO()

                document = SimpleDocTemplate(
                    pdf_buffer,
                    pagesize=A4,
                    rightMargin=45,
                    leftMargin=45,
                    topMargin=45,
                    bottomMargin=45
                )

                styles = getSampleStyleSheet()

                story = []

                story.append(
                    Paragraph(
                        "INDUSTRIAL DEFECT DETECTION",
                        styles["Title"]
                    )
                )

                story.append(
                    Paragraph(
                        "CASTING INSPECTION REPORT",
                        styles["Heading2"]
                    )
                )

                story.append(
                    Spacer(1, 20)
                )

                report_data = [
                    ["Inspection Detail", "Value"],
                    [
                        "Date and Time",
                        str(st.session_state.inspection_time)
                    ],
                    [
                        "Image Name",
                        str(st.session_state.image_name)
                    ],
                    [
                        "Prediction",
                        str(st.session_state.prediction)
                    ],
                    [
                        "Confidence",
                        f"{st.session_state.confidence:.2f}%"
                    ],
                    [
                        "Model",
                        "ResNet18"
                    ]
                ]

                report_table = Table(
                    report_data,
                    colWidths=[1.7 * inch, 4.3 * inch]
                )

                report_table.setStyle(
                    TableStyle([
                        (
                            "BACKGROUND",
                            (0, 0),
                            (-1, 0),
                            colors.HexColor("#14213D")
                        ),
                        (
                            "TEXTCOLOR",
                            (0, 0),
                            (-1, 0),
                            colors.white
                        ),
                        (
                            "GRID",
                            (0, 0),
                            (-1, -1),
                            0.5,
                            colors.grey
                        ),
                        (
                            "PADDING",
                            (0, 0),
                            (-1, -1),
                            8
                        ),
                        (
                            "VALIGN",
                            (0, 0),
                            (-1, -1),
                            "TOP"
                        )
                    ])
                )

                story.append(report_table)

                story.append(
                    Spacer(1, 20)
                )

                story.append(
                    Paragraph(
                        "Inspected Casting Image",
                        styles["Heading2"]
                    )
                )

                image_buffer = BytesIO(
                    st.session_state.image_bytes
                )

                pdf_image = PDFImage(
                    image_buffer,
                    width=4.5 * inch,
                    height=3.5 * inch,
                    kind="proportional"
                )

                story.append(pdf_image)

                story.append(
                    Spacer(1, 20)
                )

                story.append(
                    Paragraph(
                        "Note: This result is generated by a trained "
                        "AI model. Verify important inspection "
                        "decisions independently.",
                        styles["BodyText"]
                    )
                )

                document.build(story)

                pdf_buffer.seek(0)

                st.download_button(
                    label="Download Inspection PDF",
                    data=pdf_buffer.getvalue(),
                    file_name="inspection_report.pdf",
                    mime="application/pdf"
                )

            except Exception as error:

                st.error(
                    f"Could not generate the PDF report: {error}"
                )


# ---------------- RESET APPLICATION ----------------

st.divider()

if st.button("RESET / CHECK ANOTHER IMAGE"):

    st.session_state.image_bytes = None
    st.session_state.image_name = None
    st.session_state.prediction = None
    st.session_state.confidence = None
    st.session_state.inspection_time = None

    st.rerun()


# ---------------- FOOTER ----------------

st.markdown("""
<div style="background-color:#14213D;
            color:white;
            padding:12px;
            text-align:center;
            border-radius:8px;">
    INDUSTRIAL DEFECT DETECTION SYSTEM | RESNET18
</div>
""", unsafe_allow_html=True)