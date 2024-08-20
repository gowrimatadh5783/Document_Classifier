## Document Classification using LayoutLMv3 and PyTorch Lightning

Welcome to the Document Classification project repository. This project leverages the `LayoutLMv3` model, fine-tuned using `PyTorch Lightning`, for classifying documents into categories such as Income Statements, Balance Sheets, Cash Flows, Notes, and Others. Below is the detailed documentation to help you understand and utilize the repository effectively.

### Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Features](#features)
5. [How to Use](#how-to-use)
6. [Model and Processor Details](#model-and-processor-details)

---

### Overview

This repository provides:
- **Fine-tuned `LayoutLMv3` Model:** Specifically adapted for document classification.
- **User Interface (UI):** A simple and intuitive UI for uploading and classifying documents.
- **Preprocessing Tools:** For document image preprocessing and OCR (Optical Character Recognition).
- **Training and Evaluation Scripts:** For model training, validation, and testing.

---

### Prerequisites

Before you start, ensure you have the following installed:
- Python 3.8 or later
- PyTorch
- PyTorch Lightning
- Transformers (`transformers` library)
- OpenCV
- Tesseract OCR

---

### Installation

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/your-username/document-classification.git
    cd document-classification
    ```

2. **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3. **Set Up Tesseract OCR:**

    - Ensure Tesseract is installed and accessible via command line.

---

### Features

- **Document Classification:** Classify documents into predefined categories using a fine-tuned `LayoutLMv3` model.
- **UI Interface:** A user-friendly interface for uploading and classifying documents.
- **Preprocessing Pipelines:** Integrated tools for image preprocessing and text extraction using Tesseract OCR.

---

### How to Use

1. **UI-based Classification:**

    - Launch the UI by running the following command:
      
      ```bash
      python app.py
      ```
      
    - Upload your document, and the model will classify it into one of the predefined categories.

2. **Script-based Classification:**

    - To classify documents via scripts, use the provided scripts in the `scripts/` folder.
      
      Example command:
      
      ```bash
      python scripts/classify_document.py --input_path your-document-path
      ```

---

### Model and Processor Details

- **Model Used:** `LayoutLMv3`
- **Training Framework:** PyTorch Lightning
- **OCR Processor:** Tesseract OCR for text extraction
- **Preprocessing:** OpenCV for image processing tasks

---

This documentation provides the essential steps to get started with the Document Classification project. If you have any issues or need further guidance, feel free to explore the code and comments within the repository.
