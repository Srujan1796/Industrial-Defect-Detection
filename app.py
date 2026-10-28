import csv
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import torch
import torch.nn as nn
from torchvision import models, transforms
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import os

# ---------------- MODEL SETUP ----------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR, "industrial_defect_model_final.pth"
)

HISTORY_FILE = os.path.join(
    BASE_DIR, "prediction_history.csv"
)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}\n"
        "Keep app.py and the model file in the same folder."
    )

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

classes = ["def_front", "ok_front"]

# ---------------- IMAGE TRANSFORM ----------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------------- COLOURS ----------------

NAVY = "#14213D"
BLUE = "#1769AA"
BLUE_ACTIVE = "#0D4F85"
GREEN = "#187A3D"
GREEN_ACTIVE = "#105A2B"
DARK_GREY = "#374151"
PURPLE = "#7C3AED"
PURPLE_ACTIVE = "#5B21B6"
LIGHT_BG = "#E8EDF3"
WHITE = "#FFFFFF"
TEXT = "#172033"
MUTED = "#526174"
RED = "#B42318"
BORDER = "#C5D0DC"
ORANGE = "#B45309"
ORANGE_ACTIVE = "#92400E"

# ---------------- SAVE PREDICTION HISTORY ----------------

def save_history(filename, prediction, confidence):
    """Save inspection results to a CSV file."""

    file_exists = os.path.exists(HISTORY_FILE)

    with open(
        HISTORY_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        if not file_exists or os.path.getsize(HISTORY_FILE) == 0:
            writer.writerow([
                "Date and Time",
                "Image Name",
                "Prediction",
                "Confidence"
            ])

        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            filename,
            prediction,
            f"{confidence:.2f}%"
        ])


# ---------------- MAIN WINDOW ----------------

root = tk.Tk()
root.title("Industrial Defect Detection")
root.configure(bg=LIGHT_BG)

try:
    root.state("zoomed")
except tk.TclError:
    root.geometry(
        f"{root.winfo_screenwidth()}x"
        f"{root.winfo_screenheight()}+0+0"
    )

root.minsize(900, 650)

image_path = None
photo = None
original_image = None

# ---------------- HEADER ----------------

header = tk.Frame(root, bg=NAVY, height=100)
header.pack(fill="x")
header.pack_propagate(False)

tk.Label(
    header,
    text="INDUSTRIAL DEFECT DETECTION",
    font=("Arial", 25, "bold"),
    bg=NAVY,
    fg=WHITE
).pack(pady=(18, 5))

tk.Label(
    header,
    text="AI-POWERED CASTING INSPECTION | RESNET18",
    font=("Arial", 12, "bold"),
    bg=NAVY,
    fg="#DCE7F5"
).pack()

# ---------------- MAIN CONTENT ----------------

content = tk.Frame(root, bg=LIGHT_BG)
content.pack(fill="both", expand=True, padx=20, pady=18)

content.grid_columnconfigure(0, weight=3, uniform="column")
content.grid_columnconfigure(1, weight=2, uniform="column")
content.grid_rowconfigure(0, weight=1)

# ---------------- IMAGE PANEL ----------------

image_panel = tk.Frame(
    content,
    bg=WHITE,
    highlightbackground=BORDER,
    highlightthickness=1
)

image_panel.grid(
    row=0,
    column=0,
    sticky="nsew",
    padx=(0, 10)
)

image_panel.grid_rowconfigure(1, weight=1)
image_panel.grid_columnconfigure(0, weight=1)

tk.Label(
    image_panel,
    text="CASTING IMAGE PREVIEW",
    font=("Arial", 17, "bold"),
    bg=WHITE,
    fg=NAVY
).grid(row=0, column=0, pady=15)

preview_area = tk.Frame(image_panel, bg="#F4F7FA")

preview_area.grid(
    row=1,
    column=0,
    sticky="nsew",
    padx=18,
    pady=(0, 10)
)

preview_area.grid_rowconfigure(0, weight=1)
preview_area.grid_columnconfigure(0, weight=1)

image_label = tk.Label(
    preview_area,
    text="No image uploaded\n\nClick the Upload Image button below",
    font=("Arial", 16, "bold"),
    bg="#F4F7FA",
    fg=MUTED,
    justify="center"
)

image_label.grid(row=0, column=0, sticky="nsew")

filename_label = tk.Label(
    image_panel,
    text="Supported formats: JPG, JPEG, PNG",
    font=("Arial", 11),
    bg=WHITE,
    fg=MUTED,
    wraplength=700
)

filename_label.grid(
    row=2,
    column=0,
    pady=(0, 15),
    padx=10
)

# ---------------- RESULT PANEL ----------------

result_panel = tk.Frame(
    content,
    bg=WHITE,
    highlightbackground=BORDER,
    highlightthickness=1
)

result_panel.grid(
    row=0,
    column=1,
    sticky="nsew",
    padx=(10, 0)
)

tk.Label(
    result_panel,
    text="INSPECTION CONTROL",
    font=("Arial", 17, "bold"),
    bg=WHITE,
    fg=NAVY
).pack(pady=(22, 8))

tk.Label(
    result_panel,
    text="Upload a casting image and run the model.",
    font=("Arial", 12),
    bg=WHITE,
    fg=MUTED,
    wraplength=350,
    justify="center"
).pack(pady=(0, 20), padx=15)

# ---------------- RESULT DISPLAY ----------------

result_box = tk.Frame(
    result_panel,
    bg="#F4F7FA",
    highlightbackground=BORDER,
    highlightthickness=1
)

result_box.pack(fill="x", padx=18, pady=10)

tk.Label(
    result_box,
    text="PREDICTION",
    font=("Arial", 12, "bold"),
    bg="#F4F7FA",
    fg=MUTED
).pack(pady=(20, 8))

result_label = tk.Label(
    result_box,
    text="WAITING FOR IMAGE",
    font=("Arial", 19, "bold"),
    bg="#F4F7FA",
    fg=TEXT,
    wraplength=350,
    justify="center"
)

result_label.pack(padx=10, pady=10)

confidence_label = tk.Label(
    result_box,
    text="Confidence: --",
    font=("Arial", 14, "bold"),
    bg="#F4F7FA",
    fg=TEXT
)

confidence_label.pack(pady=(5, 20))

# ---------------- IMAGE PREVIEW ----------------

def show_preview():
    """Resize the image to fit the preview area."""

    global photo

    if original_image is None:
        return

    preview_area.update_idletasks()

    max_width = max(preview_area.winfo_width() - 30, 300)
    max_height = max(preview_area.winfo_height() - 30, 250)

    preview = original_image.copy()

    preview.thumbnail(
        (max_width, max_height),
        Image.Resampling.LANCZOS
    )

    photo = ImageTk.PhotoImage(preview)

    image_label.configure(
        image=photo,
        text="",
        bg="#F4F7FA"
    )


# ---------------- UPLOAD IMAGE ----------------

def upload_image():
    global image_path, original_image, photo

    file_path = filedialog.askopenfilename(
        title="Select Casting Image",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png"),
            ("All files", "*.*")
        ]
    )

    if not file_path:
        return

    try:
        original_image = Image.open(file_path).convert("RGB")
        image_path = file_path

        show_preview()

        filename_label.configure(
            text=os.path.basename(file_path),
            fg=TEXT
        )

        # Clear the previous prediction when a new image is selected.
        result_label.configure(
            text="IMAGE READY",
            fg=BLUE
        )

        confidence_label.configure(
            text="Confidence: --",
            fg=TEXT
        )

    except Exception as error:
        messagebox.showerror(
            "Image Error",
            f"Unable to open this image.\n{error}"
        )


# ---------------- PREDICT DEFECT ----------------

def predict_image():
    if image_path is None or original_image is None:
        messagebox.showwarning(
            "No Image",
            "Please upload an image first."
        )
        return

    try:
        input_tensor = transform(original_image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        predicted_class = classes[predicted.item()]
        confidence_percent = confidence.item() * 100

        if predicted_class == "def_front":
            prediction_text = "Defective Casting"

            result_label.configure(
                text="DEFECTIVE CASTING",
                fg=RED
            )
        else:
            prediction_text = "Normal Casting"

            result_label.configure(
                text="NORMAL CASTING",
                fg=GREEN
            )

        confidence_label.configure(
            text=f"Confidence: {confidence_percent:.2f}%",
            fg=TEXT
        )

        save_history(
            os.path.basename(image_path),
            prediction_text,
            confidence_percent
        )

    except Exception as error:
        messagebox.showerror(
            "Prediction Error",
            f"Prediction failed.\n{error}"
        )


# ---------------- RESET APPLICATION ----------------

def reset_app():
    global image_path, original_image, photo

    image_path = None
    original_image = None
    photo = None

    image_label.configure(
        image="",
        text="No image uploaded\n\nClick the Upload Image button below",
        bg="#F4F7FA",
        fg=MUTED
    )

    filename_label.configure(
        text="Supported formats: JPG, JPEG, PNG",
        fg=MUTED
    )

    result_label.configure(
        text="WAITING FOR IMAGE",
        fg=TEXT
    )

    confidence_label.configure(
        text="Confidence: --",
        fg=TEXT
    )


# ---------------- VIEW PREDICTION HISTORY ----------------

def view_history():
    history_window = tk.Toplevel(root)
    history_window.title("Prediction History")
    history_window.geometry("950x450")
    history_window.configure(bg=LIGHT_BG)

    tk.Label(
        history_window,
        text="PREDICTION HISTORY",
        font=("Arial", 20, "bold"),
        bg=NAVY,
        fg=WHITE,
        pady=15
    ).pack(fill="x")

    table_frame = tk.Frame(history_window, bg=WHITE)

    table_frame.pack(
        fill="both",
        expand=True,
        padx=15,
        pady=15
    )

    columns = (
        "Date and Time",
        "Image Name",
        "Prediction",
        "Confidence"
    )

    table = ttk.Treeview(
        table_frame,
        columns=columns,
        show="headings"
    )

    widths = {
        "Date and Time": 190,
        "Image Name": 260,
        "Prediction": 220,
        "Confidence": 150
    }

    for column in columns:
        table.heading(column, text=column)
        table.column(
            column,
            width=widths[column],
            anchor="center"
        )

    scrollbar = ttk.Scrollbar(
        table_frame,
        orient="vertical",
        command=table.yview
    )

    table.configure(yscrollcommand=scrollbar.set)

    table.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    records_found = False

    if os.path.exists(HISTORY_FILE):
        with open(
            HISTORY_FILE,
            "r",
            newline="",
            encoding="utf-8"
        ) as file:

            reader = csv.reader(file)
            next(reader, None)

            for row in reader:
                if len(row) == 4:
                    table.insert("", "end", values=row)
                    records_found = True

    if not records_found:
        tk.Label(
            history_window,
            text=(
                "No inspection records yet. "
                "Upload an image and click Predict Defect."
            ),
            font=("Arial", 12),
            bg=LIGHT_BG,
            fg=TEXT
        ).pack(pady=(0, 12))


# ---------------- COMPARE IMAGE & RESULT ----------------

def compare_image():
    """Show the uploaded image beside its AI prediction."""

    if image_path is None or original_image is None:
        messagebox.showwarning(
            "No Image",
            "Please upload an image first."
        )
        return

    prediction_text = result_label.cget("text")
    confidence_text = confidence_label.cget("text")

    if prediction_text not in [
        "DEFECTIVE CASTING",
        "NORMAL CASTING"
    ]:
        messagebox.showwarning(
            "No Prediction",
            "Please click PREDICT DEFECT first."
        )
        return

    comparison_window = tk.Toplevel(root)
    comparison_window.title("Casting Image Comparison")
    comparison_window.geometry("950x650")
    comparison_window.minsize(700, 500)
    comparison_window.configure(bg=LIGHT_BG)

    # Comparison window heading
    tk.Label(
        comparison_window,
        text="CASTING INSPECTION COMPARISON",
        font=("Arial", 20, "bold"),
        bg=NAVY,
        fg=WHITE,
        pady=15
    ).pack(fill="x")

    # Main comparison area
    comparison_frame = tk.Frame(
        comparison_window,
        bg=LIGHT_BG
    )

    comparison_frame.pack(
        fill="both",
        expand=True,
        padx=20,
        pady=20
    )

    comparison_frame.grid_columnconfigure(
        0, weight=1, uniform="compare"
    )
    comparison_frame.grid_columnconfigure(
        1, weight=1, uniform="compare"
    )
    comparison_frame.grid_rowconfigure(0, weight=1)

    # Left panel: original image
    original_panel = tk.Frame(
        comparison_frame,
        bg=WHITE,
        highlightbackground=BORDER,
        highlightthickness=1
    )

    original_panel.grid(
        row=0,
        column=0,
        sticky="nsew",
        padx=(0, 10)
    )

    tk.Label(
        original_panel,
        text="ORIGINAL CASTING IMAGE",
        font=("Arial", 14, "bold"),
        bg=WHITE,
        fg=NAVY
    ).pack(pady=15)

    preview_image = original_image.copy()

    preview_image.thumbnail(
        (400, 350),
        Image.Resampling.LANCZOS
    )

    original_photo = ImageTk.PhotoImage(preview_image)

    original_image_display = tk.Label(
        original_panel,
        image=original_photo,
        bg=WHITE
    )

    original_image_display.image = original_photo

    original_image_display.pack(
        fill="both",
        expand=True,
        padx=10,
        pady=10
    )

    tk.Label(
        original_panel,
        text=os.path.basename(image_path),
        font=("Arial", 10),
        bg=WHITE,
        fg=MUTED,
        wraplength=350
    ).pack(pady=10)

    # Right panel: prediction result
    result_panel_compare = tk.Frame(
        comparison_frame,
        bg=WHITE,
        highlightbackground=BORDER,
        highlightthickness=1
    )

    result_panel_compare.grid(
        row=0,
        column=1,
        sticky="nsew",
        padx=(10, 0)
    )

    tk.Label(
        result_panel_compare,
        text="AI INSPECTION RESULT",
        font=("Arial", 14, "bold"),
        bg=WHITE,
        fg=NAVY
    ).pack(pady=15)

    if prediction_text == "DEFECTIVE CASTING":
        result_color = RED
        result_description = (
            "The model classified this image "
            "as a defective casting."
        )
    else:
        result_color = GREEN
        result_description = (
            "The model classified this image "
            "as a normal casting."
        )

    tk.Label(
        result_panel_compare,
        text=prediction_text,
        font=("Arial", 19, "bold"),
        bg=WHITE,
        fg=result_color,
        wraplength=350
    ).pack(pady=(35, 20), padx=10)

    tk.Label(
        result_panel_compare,
        text=confidence_text,
        font=("Arial", 15, "bold"),
        bg=WHITE,
        fg=TEXT
    ).pack(pady=10)

    tk.Label(
        result_panel_compare,
        text=result_description,
        font=("Arial", 12),
        bg=WHITE,
        fg=MUTED,
        wraplength=350,
        justify="center"
    ).pack(padx=20, pady=20)

    # Safety note
    tk.Label(
        comparison_window,
        text=(
            "AI predictions should be verified before "
            "making important industrial decisions."
        ),
        font=("Arial", 10),
        bg=LIGHT_BG,
        fg=MUTED
    ).pack(pady=(0, 15))


# ---------------- GENERATE PDF REPORT ----------------

def generate_pdf_report():
    if image_path is None or original_image is None:
        messagebox.showwarning(
            "No Image",
            "Please upload an image first."
        )
        return

    prediction_text = result_label.cget("text")
    confidence_text = confidence_label.cget("text")

    if prediction_text not in [
        "DEFECTIVE CASTING",
        "NORMAL CASTING"
    ]:
        messagebox.showwarning(
            "No Prediction",
            "Please click PREDICT DEFECT before generating the report."
        )
        return

    pdf_path = filedialog.asksaveasfilename(
        title="Save Inspection Report",
        defaultextension=".pdf",
        filetypes=[("PDF files", "*.pdf")],
        initialfile="inspection_report.pdf"
    )

    if not pdf_path:
        return

    try:
        pdf = canvas.Canvas(pdf_path, pagesize=A4)
        page_width, page_height = A4

        pdf.setTitle("Industrial Defect Inspection Report")

        # Report heading
        pdf.setFillColorRGB(0.08, 0.13, 0.24)
        pdf.setFont("Helvetica-Bold", 18)
        pdf.drawCentredString(
            page_width / 2,
            page_height - 60,
            "INDUSTRIAL DEFECT DETECTION"
        )

        pdf.setFillColorRGB(0, 0, 0)
        pdf.setFont("Helvetica-Bold", 14)
        pdf.drawCentredString(
            page_width / 2,
            page_height - 85,
            "CASTING INSPECTION REPORT"
        )

        pdf.line(
            50,
            page_height - 105,
            page_width - 50,
            page_height - 105
        )

        # Inspection details
        y = page_height - 145

        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(60, y, "Inspection Details")

        y -= 30

        details = [
            (
                "Date and Time",
                datetime.now().strftime("%d-%m-%Y %H:%M:%S")
            ),
            ("Image Name", os.path.basename(image_path)),
            ("Prediction", prediction_text),
            ("Confidence", confidence_text),
            ("Model", "ResNet18")
        ]

        for label, value in details:
            pdf.setFont("Helvetica-Bold", 10)
            pdf.drawString(60, y, f"{label}:")

            pdf.setFont("Helvetica", 10)
            pdf.drawString(175, y, str(value)[:65])

            y -= 25

        # Include the inspected image
        y -= 10

        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(60, y, "Inspected Casting Image")

        image_top = y - 15
        image_box_height = 220
        image_box_width = page_width - 120

        pdf.drawImage(
            image_path,
            60,
            image_top - image_box_height,
            width=image_box_width,
            height=image_box_height,
            preserveAspectRatio=True,
            anchor="c",
            mask="auto"
        )

        # Disclaimer
        pdf.setFont("Helvetica-Oblique", 9)
        pdf.drawString(
            60,
            55,
            "Note: This result is generated by a trained AI model."
        )

        pdf.drawString(
            60,
            42,
            "Please verify important inspection decisions independently."
        )

        pdf.save()

        messagebox.showinfo(
            "Report Generated",
            f"PDF report saved successfully:\n{pdf_path}"
        )

    except Exception as error:
        messagebox.showerror(
            "PDF Error",
            f"Could not generate the report:\n{error}"
        )


# ---------------- BUTTONS ----------------

button_frame = tk.Frame(result_panel, bg=WHITE)

button_frame.pack(
    fill="x",
    padx=18,
    pady=(20, 10)
)

upload_button = tk.Button(
    button_frame,
    text="UPLOAD IMAGE",
    command=upload_image,
    font=("Arial", 14, "bold"),
    bg=BLUE,
    fg=WHITE,
    activebackground=BLUE_ACTIVE,
    activeforeground=WHITE,
    relief="flat",
    cursor="hand2",
    pady=14
)

upload_button.pack(fill="x", pady=8)

predict_button = tk.Button(
    button_frame,
    text="PREDICT DEFECT",
    command=predict_image,
    font=("Arial", 14, "bold"),
    bg=GREEN,
    fg=WHITE,
    activebackground=GREEN_ACTIVE,
    activeforeground=WHITE,
    relief="flat",
    cursor="hand2",
    pady=14
)

predict_button.pack(fill="x", pady=8)

reset_button = tk.Button(
    button_frame,
    text="RESET",
    command=reset_app,
    font=("Arial", 14, "bold"),
    bg=DARK_GREY,
    fg=WHITE,
    activebackground="#1F2937",
    activeforeground=WHITE,
    relief="flat",
    cursor="hand2",
    pady=12
)

reset_button.pack(fill="x", pady=8)

history_button = tk.Button(
    button_frame,
    text="VIEW HISTORY",
    command=view_history,
    font=("Arial", 14, "bold"),
    bg=NAVY,
    fg=WHITE,
    activebackground="#243B64",
    activeforeground=WHITE,
    relief="flat",
    cursor="hand2",
    pady=12
)

history_button.pack(fill="x", pady=8)

pdf_button = tk.Button(
    button_frame,
    text="GENERATE PDF REPORT",
    command=generate_pdf_report,
    font=("Arial", 14, "bold"),
    bg=PURPLE,
    fg=WHITE,
    activebackground=PURPLE_ACTIVE,
    activeforeground=WHITE,
    relief="flat",
    cursor="hand2",
    pady=12
)

pdf_button.pack(fill="x", pady=8)

# NEW BUTTON: COMPARE IMAGE & RESULT
compare_button = tk.Button(
    button_frame,
    text="COMPARE IMAGE & RESULT",
    command=compare_image,
    font=("Arial", 14, "bold"),
    bg=ORANGE,
    fg=WHITE,
    activebackground=ORANGE_ACTIVE,
    activeforeground=WHITE,
    relief="flat",
    cursor="hand2",
    pady=12
)

compare_button.pack(fill="x", pady=8)

tk.Label(
    result_panel,
    text="Result is based on the trained model.",
    font=("Arial", 10),
    bg=WHITE,
    fg=MUTED,
    wraplength=350
).pack(side="bottom", pady=18, padx=10)


# ---------------- RESPONSIVE IMAGE PREVIEW ----------------

def on_preview_resize(event):
    if original_image is not None:
        show_preview()


preview_area.bind("<Configure>", on_preview_resize)


# ---------------- FOOTER ----------------

footer = tk.Label(
    root,
    text="INDUSTRIAL DEFECT DETECTION SYSTEM",
    font=("Arial", 10, "bold"),
    bg=NAVY,
    fg=WHITE,
    pady=8
)

footer.pack(fill="x", side="bottom")


# ---------------- RUN APPLICATION ----------------

root.mainloop()