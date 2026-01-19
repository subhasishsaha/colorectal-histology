# Colorectal Histology Image Classification

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange)
![Gradio](https://img.shields.io/badge/Gradio-UI-yellow)

A sophisticated machine learning system for classifying colorectal cancer histology images using a hierarchical approach. This project combines handcrafted feature extraction (GLCM, Gabor filters, geometric properties) with a multi-stage classification model to accurately identify tissue types.

## 🌟 Key Features

*   **Hierarchical Classification Architecture**:
    *   **Level 1 (Superclass)**: Classifies input into broad categories: Tumour-related, Immune, Structural, or Normal.
    *   **Level 2 (Subclass)**: Refines prediction into 8 specific tissue types using specialized sub-models.
*   **Robust Feature Engineering**:
    *   **Texture**: Gray Level Co-occurrence Matrix (GLCM) - Contrast, Homogeneity, Energy, Correlation.
    *   **Frequency**: Gabor filters extracting texture information at multiple orientations and scales.
    *   **Geometry**: Shape properties (Area, Perimeter, Circularity, etc.) from segmented regions.
*   **Interactive Interface**: User-friendly web app built with **Gradio** for real-time inference on uploaded images.

## 📊 Dataset

The model is trained on the **Kather 2016** dataset, sourced via **TensorFlow Datasets (`colorectal_histology`)**.

*   **Source**: [Kather et al. (2016)](https://zenodo.org/record/53169#.YgOqN99OmUk)
*   **Content**: 5,000 histological images of human colorectal cancer.
*   **Classes** (8 distinct tissue types):
    1.  Tumour epithelium
    2.  Simple stroma
    3.  Complex stroma
    4.  Immune cell conglomerates
    5.  Debris and mucus
    6.  Mucosal glands
    7.  Adipose tissue
    8.  Background

## 📂 Project Structure

```
├── main/
│   ├── src/            # Source Jupyter notebooks for training and analysis
│   └── ...
├── models/             # Pre-trained model files (.pkl)
│   ├── heirarchial_model.pkl
│   ├── tumour_model.pkl
│   ├── structural_model.pkl
│   └── normal_model.pkl
├── ui/
│   └── app.py          # Gradio web application
├── requirements.txt    # Project dependencies
└── README.md           # Documentation
```

## 🛠️ Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/subhasishsaha/colorectal-cancer-classification.git
    cd colorectal-cancer-classification
    ```

2.  **Install Dependencies**:
    Create a virtual environment (optional but recommended) and install requirements:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Model Configuration**:
    Ensure the trained `.pkl` models are located in the `models/` directory.
    *Note: If running locally, check `ui/app.py` to ensure `joblib.load` paths point to `models/` (e.g., `./models/heirarchial_model.pkl`).*

## 🚀 Usage

Launch the interactive web interface to test the classifier:

```bash
python ui/app.py
```

Access the UI in your browser at `http://127.0.0.1:7860`.

### Workflow
1.  **Upload Image**: Select a histology tile image.
2.  **Processing**: The system segments the image using KMeans and extracts dense handcrafted features.
3.  **Prediction**: The hierarchical models predict the superclass and then the precise tissue label.

## 🧠 Methodology Details

The system avoids a "black box" approach by using interpretable handcrafted features fed into a hierarchical classifier:

1.  **Preprocessing**: RGB to LAB/Grayscale conversion and Standardization.
2.  **Segmentation**: KMeans clustering (k=3) isolates the Region of Interest (ROI).
3.  **Feature Extraction**:
    *   *GLCM*: Captures spatial relationships of pixel intensities.
    *   *Gabor Features*: Analyzes frequency content in different directions.
    *   *Region Props*: Measures physical properties of the segmented mask.
4.  **Classification Stack**:
    *   A primary classifier determines the "Superclass".
    *   Depending on the superclass, the data is routed to a specialized secondary classifier for the final output.

## 🤝 Contributing

Contributions are welcome!
*   Check `main/src/script.ipynb` for the underlying training logic and experiments.
*   The UI logic resides in `ui/app.py`.

## 📜 License

This project is open-source. Please see the [LICENSE](LICENSE) file for details.
