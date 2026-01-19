# ===========================
# 📦 Imports
# ===========================
import gradio as gr
import pickle
import numpy as np
from PIL import Image
from torchvision import transforms
import joblib
import pandas as pd

# ===========================
# ⚙️ Load Models
# ===========================
# Update paths according to your Kaggle input directory
MODEL_PATH = "/kaggle/input/your-models"

hybrid_model = joblib.load("/kaggle/input/colorectal-histology-model/scikitlearn/default/1/kaggle/working/heirarchial_model.pkl")

tumour_model = joblib.load("/kaggle/input/colorectal-histology-model/scikitlearn/default/1/kaggle/working/tumour_model.pkl")

structural_model = joblib.load("/kaggle/input/colorectal-histology-model/scikitlearn/default/1/kaggle/working/structural_model.pkl")

normal_model = joblib.load("/kaggle/input/colorectal-histology-model/scikitlearn/default/1/kaggle/working/normal_model.pkl")

# ===========================
# 🧠 Dictionaries
# ===========================
super_class_dict = {
    0: "tumour-related",
    1: "immune",
    2: "structural",
    3: "normal",
    4: "unknown"
}

label_dict = {
    0: 'tumour epithelium',
    1: 'simple stroma',
    2: 'complex stroma',
    3: 'immune cell conglomerates',
    4: 'debris and mucus',
    5: 'mucosal glands',
    6: 'adipose tissue',
    7: 'background'
}

def super_class(label):
    match label:
        case 0 | 1 | 2:
            category = 0
        case 3:
            category = 1
        case 4 | 7:
            category = 2
        case 5 | 6:
            category = 3
        case _:
            category = 4
    return category

# ===========================
# 🧩 Feature Extraction
# ===========================
import numpy as np
import cv2
from skimage.filters import gabor
from skimage.feature import graycomatrix, graycoprops
from skimage import morphology
from skimage.measure import regionprops
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def extract_features_single(image_pil, k_clusters=3, out_size=(150, 150)):
    """
    Extract the same handcrafted features for a single user-uploaded PIL image.
    Returns a numpy feature vector (1, N) ready for model prediction.
    """
    # Convert PIL to numpy
    img = np.array(image_pil)

    # Ensure correct channel order
    if img.ndim == 2:  # grayscale
        img = np.stack([img]*3, axis=-1)
    elif img.shape[-1] == 4:  # RGBA → RGB
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

    # KMeans segmentation (same as training)
    H, W, _ = lab.shape
    features = lab.reshape(-1, 3).astype(np.float32)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    km = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(features_scaled)
    segmented = labels.reshape(H, W)

    # Select ROI = darkest average gray region
    roi_cluster = np.argmin([np.mean(gray[segmented == i]) for i in range(k_clusters)])
    mask = (segmented == roi_cluster).astype(np.uint8)
    mask = morphology.remove_small_objects(mask.astype(bool), min_size=50)
    mask = mask.astype(np.uint8)

    # Resize for consistency
    gray_resized = cv2.resize(gray, out_size, interpolation=cv2.INTER_AREA)
    mask_resized = cv2.resize(mask, out_size, interpolation=cv2.INTER_NEAREST).astype(bool)

    # ---- Region features ----
    props = regionprops(mask_resized.astype(int), intensity_image=gray_resized)
    if not props:
        raise ValueError("No region found in mask.")
    region = props[0]

    features_dict = {
        "area": region.area,
        "perimeter": region.perimeter,
        "circularity": (4 * np.pi * region.area) / (region.perimeter ** 2 + 1e-6),
        "eccentricity": region.eccentricity,
        "solidity": region.solidity,
        "extent": region.extent,
        "mean_intensity": region.mean_intensity,
        "std_intensity": np.std(gray_resized[mask_resized]),
        "skewness": skew(gray_resized[mask_resized].ravel()),
        "kurtosis": kurtosis(gray_resized[mask_resized].ravel())
    }

    # ---- Texture features (GLCM) ----
    glcm = graycomatrix(gray_resized, [1], [0], symmetric=True, normed=True)
    features_dict.update({
        "contrast": graycoprops(glcm, 'contrast')[0, 0],
        "homogeneity": graycoprops(glcm, 'homogeneity')[0, 0],
        "energy": graycoprops(glcm, 'energy')[0, 0],
        "correlation": graycoprops(glcm, 'correlation')[0, 0],
    })

    # ---- Gabor features ----
    thetas = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    freqs = [0.1, 0.2, 0.3]
    feats = []
    for theta in thetas:
        for freq in freqs:
            filt_real, _ = gabor(gray_resized, frequency=freq, theta=theta)
            filt_real = (filt_real - filt_real.min()) / (filt_real.max() - filt_real.min() + 1e-6)
            feats.append(filt_real)
    gabor_stack = np.stack(feats, axis=-1)
    roi_vals = gabor_stack[mask_resized]
    per_filter_mean = roi_vals.mean(axis=0)
    per_filter_std = roi_vals.std(axis=0)
    features_dict.update({
        "gabor_mean": float(np.mean(per_filter_mean)),
        "gabor_std": float(np.mean(per_filter_std))
    })

    # ---- Convert to vector (1 x N) ----
    feature_vector = np.array(list(features_dict.values()), dtype=np.float32).reshape(1, -1)
    return feature_vector



# ===========================
# 🚀 Prediction Logic
# ===========================
def predict(image):
    # Extract features
    try:
        image_features =  extract_features_single(image)
    except:
        return "unknown", "unknown"
    features=pd.DataFrame(image_features)
    # Predict superclass
    super_pred = hybrid_model.predict(features)[0]
    superclass_name = super_class_dict.get(super_pred, "unknown")
    print(superclass_name)

    # Based on superclass, pick correct model
    if super_pred == 0:  # tumour-related
        sub_pred = tumour_model.predict(features)[0][0]
        print(sub_pred)
        label_name = label_dict.get(sub_pred,"unknown")

    elif super_pred == 1:  # immune
        label_name = "immune cell conglomerates"

    elif super_pred == 2:  # structural
        sub_pred = structural_model.predict(features)[0]
        label_name = label_dict.get(sub_pred,"unknown")

    elif super_pred == 3:  # normal
        sub_pred = normal_model.predict(features)[0]
        label_name = label_dict.get(sub_pred,"unknown")

    else:  # unknown
        label_name = "unknown"

    return superclass_name, label_name

# ===========================
# 🎨 Gradio UI
# ===========================
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Histopathology Image"),
    outputs=[
        gr.Label(label="Superclass Prediction"),
        gr.Label(label="Detailed Label Prediction")
    ],
    title="🧬 Hybrid Image Classifier",
    description=(
        "Uploads an image → Extracts features → Predicts superclass (0–4) "
        "and then uses the corresponding submodel for fine-grained classification."
    )
)

# ===========================
# ▶️ Launch
# ===========================
demo.launch()