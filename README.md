# Colorectal Histology Image Classification

A hybrid machine learning system designed to classify colorectal cancer histology images. This project leverages handcrafted feature extraction (GLCM, Gabor filters, geometric properties) combined with a hierarchical classification model to distinguish between various tissue types.

## 🌟 Key Features

*   **Hierarchical Classification**: Uses a two-stage approach:
    1.  **Superclass Prediction**: Classifies tissue into broad categories (Tumour-related, Immune, Structural, Normal).
    2.  **Subclass Prediction**: Refines the classification into specific tissue types (e.g., Tumour Epithelium, Simple Stroma, Adipose Tissue) using specialized models.
*   **Handcrafted Feature Extraction**: robust feature engineering pipeline including:
    *   **Texture Analysis**: GLCM (Contrast, Homogeneity, Energy, Correlation).
    *   **Frequency Analysis**: Gabor filters at multiple orientations and frequencies.
    *   **Geometric Properties**: Area, Perimeter, Circularity, Eccentricity, etc., from segmented regions.
*   **Interactive UI**: A user-friendly web interface built with **Gradio** for easy image uploading and real-time prediction.

## 📂 Project Structure

```
├── main/
│   ├── src/            # Core source logic (Jupyter notebooks)
│   └── ...
├── models/             # Directory for storing trained model files (.pkl)
├── ui/
│   └── app.py          # Gradio application entry point
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

## 🛠️ Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd colorectal-cancer-classification
    ```

2.  **Install Dependencies**:
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Model Setup**:
    Ensure your trained model files (`heirarchial_model.pkl`, `tumour_model.pkl`, etc.) are placed in the expected directory.
    *Note: The default paths in `ui/app.py` are set to `/kaggle/input/...`. You may need to update `MODEL_PATH` or the specific `joblib.load` paths in `ui/app.py` to match your local `models/` directory.*

## 🚀 Usage

To start the interactive web interface:

```bash
python ui/app.py
```

This will launch a local server (usually at `http://127.0.0.1:7860`). Open this URL in your browser to interact with the classifier.

## 🧠 Model Logic

The system operates on a hierarchical basis:
1.  **Input**: Histology image tile.
2.  **Processing**: Color conversion (RGB->LAB/Gray), KMeans segmentation, and feature extraction.
3.  **Level 1**: The **Hybrid Model** predicts the "Superclass".
4.  **Level 2**: Based on the superclass, the image is passed to a specialized sub-model (Tumour, Structural, or Normal model) for the final label.

## 🤝 Contributing

Contributions are welcome! Please feel free to inspect `main/src/script.ipynb` for the training logic and `ui/app.py` for the inference pipeline.
