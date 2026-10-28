import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from PIL import Image
from datetime import datetime
from io import BytesIO
import pandas as pd

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as PDFImage,
    Table,
    TableStyle
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Industrial Defect Detection",
    page_icon="🏭",
    layout="wide"
)


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown(
    """
    <style>
    .main-title {
        text-align: center;
        color: #1565C0;
        font-size: 36px;
        font-weight: bold;
    }

    .subtitle {
        text-align: center;
        color: #666666;
        font-size: 17px;
    }

    .result-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #F0F2F6;
        text-align: center;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# MODEL CONFIGURATION
# --------------------------------------------------

MODEL_PATH = "industrial_defect_model_final.pth"

CLASS_NAMES = [
    "def_front",
    "ok_front"
]

DEVICE = torch.device("cpu")


# --------------------------------------------------
# IMAGE TRANSFORMATIONS
# Keep the same preprocessing used during training.
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():

    model = models.resnet18(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        2
    )

    state_dict = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()

    return model


# --------------------------------------------------
# INITIALIZE SESSION STATE
# --------------------------------------------------

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "confidence" not in st.session_state:
    st.session_state.confidence = None

if "inspection_time" not in st.session_state:
    st.session_state.inspection_time = None

if "image_name" not in st.session_state:
    st.session_state.image_name = None

if "image_bytes" not in st.session_state:
    st.session_state.image_bytes = None

if "history" not in st.session_state:
    st.session_state.history = []

if "uploader_version" not in st.session_state:
    st.session_state.uploader_version = 0


# --------------------------------------------------
# PDF REPORT GENERATION
# --------------------------------------------------

def generate_pdf_report():

    pdf_buffer = BytesIO()

    document = SimpleDocTemplate(
        pdf_buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    story = []

    story.append(
        Paragraph(
            "Industrial Defect Detection Report",
            styles["Title"]
        )
    )

    story.append(Spacer(1, 20))

    story.append(
        Paragraph(
            "Inspection Details",
            styles["Heading2"]
        )
    )

    story.append(Spacer(1, 10))

    details = [
        ["Image Name", str(st.session_state.image_name)],
        [
            "Inspection Date",
            str(st.session_state.inspection_time)
        ],
        [
            "Prediction",
            str(st.session_state.prediction)
        ],
        [
            "Confidence",
            f"{st.session_state.confidence:.2f}%"
        ]
    ]

    details_table = Table(
        details,
        colWidths=[150, 300]
    )

    details_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (0, -1),
                colors.lightgrey
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.grey
            ),
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE"
            ),
            (
                "LEFTPADDING",
                (0, 0),
                (-1, -1),
                8
            ),
            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                8
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                8
            )
        ])
    )

    story.append(details_table)
    story.append(Spacer(1, 25))

    story.append(
        Paragraph(
            "Inspected Image",
            styles["Heading2"]
        )
    )

    story.append(Spacer(1, 10))

    image_stream = BytesIO(
        st.session_state.image_bytes
    )

    report_image = PDFImage(
        image_stream,
        width=4 * inch,
        height=3 * inch,
        kind="proportional"
    )

    story.append(report_image)
    story.append(Spacer(1, 20))

    story.append(
        Paragraph(
            "This report contains the result generated by "
            "the trained image classification model. "
            "The prediction should be verified when "
            "required by the inspection process.",
            styles["BodyText"]
        )
    )

    document.build(story)

    pdf_buffer.seek(0)

    return pdf_buffer.getvalue()


# --------------------------------------------------
# APPLICATION HEADER
# --------------------------------------------------

st.markdown(
    '<div class="main-title">'
    '🏭 Industrial Defect Detection'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-Based Casting Image Inspection System'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

try:

    model = load_model()

except Exception as error:

    st.error(
        f"Unable to load the trained model: {error}"
    )

    st.info(
        "Check that industrial_defect_model_final.pth "
        "exists in the project directory and that the "
        "required packages are installed."
    )

    st.stop()


# --------------------------------------------------
# IMAGE UPLOAD SECTION
# --------------------------------------------------

st.subheader("1. Upload Casting Image")

uploaded_file = st.file_uploader(
    "Choose an image to inspect",
    type=["jpg", "jpeg", "png"],
    key=f"image_uploader_{st.session_state.uploader_version}"
)


# --------------------------------------------------
# HANDLE UPLOADED IMAGE
# --------------------------------------------------

if uploaded_file is not None:

    uploaded_bytes = uploaded_file.getvalue()

    # Clear the previous prediction when a new image
    # is selected.
    if uploaded_bytes != st.session_state.image_bytes:

        st.session_state.prediction = None
        st.session_state.confidence = None
        st.session_state.inspection_time = None

        st.session_state.image_bytes = uploaded_bytes
        st.session_state.image_name = uploaded_file.name

    try:

        original_image = Image.open(
            BytesIO(uploaded_bytes)
        ).convert("RGB")

        upload_left, upload_right = st.columns(2)

        with upload_left:

            st.markdown("### Uploaded Image")

            st.image(
                original_image,
                caption=uploaded_file.name,
                use_container_width=True
            )

        with upload_right:

            st.markdown("### Image Information")

            st.write(
                f"**File name:** {uploaded_file.name}"
            )

            st.write(
                f"**Image width:** {original_image.width} pixels"
            )

            st.write(
                f"**Image height:** {original_image.height} pixels"
            )

            st.write(
                f"**Image format:** {original_image.format or 'Converted to RGB'}"
            )

        st.divider()

        # --------------------------------------------------
        # PREDICTION BUTTON
        # --------------------------------------------------

        if st.button(
            "🔍 Predict Defect",
            type="primary",
            use_container_width=True
        ):

            try:

                with st.spinner(
                    "Analyzing the casting image..."
                ):

                    input_tensor = transform(
                        original_image
                    ).unsqueeze(0).to(DEVICE)

                    with torch.no_grad():

                        outputs = model(input_tensor)

                        probabilities = torch.softmax(
                            outputs,
                            dim=1
                        )

                        confidence, predicted_class = (
                            torch.max(
                                probabilities,
                                dim=1
                            )
                        )

                    predicted_index = (
                        predicted_class.item()
                    )

                    confidence_percent = (
                        confidence.item() * 100
                    )

                    if predicted_index == 0:

                        prediction_text = (
                            "DEFECTIVE CASTING"
                        )

                    else:

                        prediction_text = (
                            "NORMAL CASTING"
                        )

                    inspection_time = (
                        datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        )
                    )

                    # Save the current prediction.
                    st.session_state.prediction = (
                        prediction_text
                    )

                    st.session_state.confidence = (
                        confidence_percent
                    )

                    st.session_state.inspection_time = (
                        inspection_time
                    )

                    # Save the inspection to session history.
                    st.session_state.history.append({
                        "Date and Time": inspection_time,
                        "Image Name": (
                            st.session_state.image_name
                        ),
                        "Prediction": prediction_text,
                        "Confidence": (
                            f"{confidence_percent:.2f}%"
                        )
                    })

            except Exception as error:

                st.error(
                    f"Prediction failed: {error}"
                )

    except Exception as error:

        st.error(
            f"Unable to open the uploaded image: {error}"
        )


# --------------------------------------------------
# DISPLAY INSPECTION RESULT
# --------------------------------------------------

if st.session_state.prediction is not None:

    st.divider()

    st.subheader("2. Inspection Result")

    result_left, result_right = st.columns(2)

    with result_left:

        st.markdown("### Prediction")

        if (
            st.session_state.prediction
            == "DEFECTIVE CASTING"
        ):

            st.error(
                "⚠️ DEFECTIVE CASTING"
            )

        else:

            st.success(
                "✅ NORMAL CASTING"
            )

        st.metric(
            "Model Confidence",
            f"{st.session_state.confidence:.2f}%"
        )

        st.write(
            f"**Inspection time:** "
            f"{st.session_state.inspection_time}"
        )

        st.write(
            f"**Image name:** "
            f"{st.session_state.image_name}"
        )

    with result_right:

        st.markdown("### Inspection Summary")

        if (
            st.session_state.prediction
            == "DEFECTIVE CASTING"
        ):

            st.warning(
                "The model classified this image as "
                "a defective casting. Review the image "
                "before making a final quality decision."
            )

        else:

            st.success(
                "The model classified this image as "
                "a normal casting."
            )

        st.caption(
            "Confidence represents the model's predicted "
            "probability for the selected class. It does "
            "not guarantee that the classification is correct."
        )


# --------------------------------------------------
# IMAGE COMPARISON
# --------------------------------------------------

if (
    st.session_state.prediction is not None
    and st.session_state.image_bytes is not None
):

    st.divider()

    st.subheader("3. Compare Image and Result")

    compare_left, compare_right = st.columns(2)

    with compare_left:

        st.markdown("### Original Image")

        st.image(
            Image.open(
                BytesIO(
                    st.session_state.image_bytes
                )
            ).convert("RGB"),
            use_container_width=True
        )

    with compare_right:

        st.markdown("### Prediction Result")

        st.markdown(
            f"**{st.session_state.prediction}**"
        )

        st.metric(
            "Confidence",
            f"{st.session_state.confidence:.2f}%"
        )


# --------------------------------------------------
# PREDICTION HISTORY
# --------------------------------------------------

st.divider()

st.subheader("4. Prediction History")

if st.session_state.history:

    history_df = pd.DataFrame(
        st.session_state.history
    )

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )

    csv_data = history_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="⬇️ Download History as CSV",
        data=csv_data,
        file_name="prediction_history.csv",
        mime="text/csv"
    )

else:

    st.info(
        "No inspections have been performed in this session yet."
    )


# --------------------------------------------------
# PDF REPORT
# --------------------------------------------------

st.divider()

st.subheader("5. Generate PDF Report")

if st.session_state.prediction is not None:

    try:

        pdf_data = generate_pdf_report()

        st.download_button(
            label="📄 Download Inspection Report (PDF)",
            data=pdf_data,
            file_name="industrial_defect_report.pdf",
            mime="application/pdf"
        )

    except Exception as error:

        st.error(
            f"Unable to generate PDF report: {error}"
        )

else:

    st.info(
        "Upload an image and click Predict Defect "
        "before generating a PDF report."
    )


# --------------------------------------------------
# RESET BUTTON
# --------------------------------------------------

st.divider()

if st.button(
    "🔄 Reset Current Inspection",
    use_container_width=True
):

    st.session_state.prediction = None
    st.session_state.confidence = None
    st.session_state.inspection_time = None
    st.session_state.image_name = None
    st.session_state.image_bytes = None

    # Change the uploader key so the previous file
    # is removed from the upload widget.
    st.session_state.uploader_version += 1

    st.rerun()


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Industrial Defect Detection | "
    "ResNet18 Deep Learning Model | "
    "Academic Project"
)