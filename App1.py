import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk, ImageDraw, ImageFont
import torch
from transformers import LayoutLMv3ForSequenceClassification, LayoutLMv3Processor
import json
from pathlib import Path
import tarfile
import os
import pickle
import easyocr
import numpy as np
from datetime import datetime

# Path to your model.tar.gz and processor/tokenizer files
model_tar_path = "/home/dbtez/Downloads/DocClassifier/DocClassifier/model.tar.gz"
extract_path = "/home/dbtez/Downloads/DocClassifier/DocClassifier/extracted_model"
processor_pkl_path = "/home/dbtez/Downloads/DocClassifier/DocClassifier/processor.pkl"
tokenizer_pkl_path = "/home/dbtez/Downloads/DocClassifier/DocClassifier/tokenizer.pkl"

# Path to your company logo image
logo_path = "/home/dbtez/Downloads/DocClassifier/DocClassifier/DBTEZ-Logo-TM-WB.png"

# Extract the tar.gz file if not already extracted
if not os.path.exists(extract_path):
    with tarfile.open(model_tar_path, "r:gz") as tar:
        tar.extractall(path=extract_path)
    print(f"Model extracted to {extract_path}")

# Define device
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load the model
model_dir = os.path.join(extract_path, "content", "best-model")
model = LayoutLMv3ForSequenceClassification.from_pretrained(model_dir).eval().to(DEVICE)

# Load the processor and tokenizer from .pkl files
with open(processor_pkl_path, "rb") as f:
    processor = pickle.load(f)

with open(tokenizer_pkl_path, "rb") as f:
    tokenizer = pickle.load(f)

# Initialize EasyOCR Reader
reader = easyocr.Reader(['en'])

def scale_bounding_box(bbox, width_scale, height_scale):
    return [
        int(bbox[0] * width_scale),
        int(bbox[1] * height_scale),
        int(bbox[2] * width_scale),
        int(bbox[3] * height_scale),
    ]

def create_bounding_box(bbox_data):
    xs = []
    ys = []
    for x, y in bbox_data:
        xs.append(x)
        ys.append(y)

    left = int(min(xs))
    top = int(min(ys))
    right = int(max(xs))
    bottom = int(max(ys))

    return [left, top, right, bottom]

def ocr_on_image(image_path):
    ocr_result = reader.readtext(str(image_path), batch_size=16)
    ocr_page = []
    for bbox, word, confidence in ocr_result:
        ocr_page.append({
            "word": word, "bounding_box": create_bounding_box(bbox)
        })
    return ocr_page

def predict_document_image(image_path: Path, model: LayoutLMv3ForSequenceClassification, processor: LayoutLMv3Processor):
    json_path = image_path.with_suffix(".json")
    
    if json_path.exists():
        # Use pre-existing .json file
        with json_path.open("r") as f:
            ocr_result = json.load(f)
    else:
        # Perform OCR on the image
        ocr_result = ocr_on_image(image_path)
    
    if not ocr_result:
        return "No OCR data available"

    with Image.open(image_path).convert("RGB") as image:
        width, height = image.size
        width_scale = 1000 / width
        height_scale = 1000 / height
        words = []
        boxes = []
        for row in ocr_result:
            boxes.append(
                scale_bounding_box(
                    row["bounding_box"],
                    width_scale,
                    height_scale
                )
            )
            words.append(row["word"])
        encoding = processor(
            image,
            words,
            boxes=boxes,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
    
    with torch.no_grad():
        output = model(
            input_ids=encoding["input_ids"].to(DEVICE),
            attention_mask=encoding["attention_mask"].to(DEVICE),
            bbox=encoding["bbox"].to(DEVICE),
            pixel_values=encoding["pixel_values"].to(DEVICE)
        )
    predicted_class = output.logits.argmax()
    return model.config.id2label[predicted_class.item()]

def select_files():
    file_paths = filedialog.askopenfilenames(filetypes=[("Image files", "*.jpg"), ("Image files", "*.jpeg"), ("Image files", "*.png")])
    for file_path in file_paths:
        file_path = Path(file_path)
        result = predict_document_image(file_path, model, processor)
        upload_date = datetime.fromtimestamp(os.path.getmtime(file_path)).strftime('%Y-%m-%d %H:%M:%S')
        table.insert("", "end", values=(file_path.name, result, upload_date))

# Main Tkinter window
root = tk.Tk()
root.title("Document Classification")
root.geometry("800x600")  # Adjusted the window size

# Create a style for the Treeview
style = ttk.Style()
style.configure("Treeview", font=("Arial", 12), rowheight=25, borderwidth=1, relief="solid")
style.configure("Treeview.Heading", font=("Arial", 14, "bold"))

# Create a frame for instructions and buttons
top_frame = tk.Frame(root, pady=10)
top_frame.pack(side="top", fill="x")

# Instructions
instructions = tk.Label(top_frame, text="Select images to classify:", font=("Arial", 14, "bold"))
instructions.pack(pady=10)

# Open button
open_button = tk.Button(top_frame, text="Select Images", command=select_files, font=("Arial", 12), bg="#4CAF50", fg="white", relief="raised")
open_button.pack(pady=5)

# Create a frame for the table and scrollbar
table_frame = tk.Frame(root, width=760, height=400, borderwidth=2, relief="solid")
table_frame.pack(fill="both", expand=True, padx=20, pady=10)

# Create a table to display results
columns = ("Filename", "Prediction", "Upload Date")
table = ttk.Treeview(table_frame, columns=columns, show="headings", style="Treeview")
table.heading("Filename", text="Filename")
table.heading("Prediction", text="Prediction")
table.heading("Upload Date", text="Upload Date")
table.column("Filename", anchor="w", width=250)
table.column("Prediction", anchor="w", width=250)
table.column("Upload Date", anchor="w", width=200)

# Add a vertical scrollbar to the table
scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=table.yview)
table.configure(yscrollcommand=scrollbar.set)
scrollbar.pack(side="right", fill="y")
table.pack(fill="both", expand=True)

# Load and display the logo
logo_image = Image.open(logo_path)
logo_image = logo_image.resize((100, 100), Image.LANCZOS)  # Resize as needed
logo_photo = ImageTk.PhotoImage(logo_image)

logo_frame = tk.Frame(root, bg="white", borderwidth=0, relief="solid")
logo_frame.pack(side="bottom", anchor="se", padx=80, pady=10)



# Run the main loop
root.mainloop()
