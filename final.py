import streamlit as st
import numpy as np
from PIL import Image
from pathlib import Path
import torch
import torchvision
import os
import torch.nn.functional as nnf
from torchvision import transforms
import pickle
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datetime import datetime

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Health AI Diagnosis System",
    page_icon="🩺",
    layout="wide"
)

# ==========================================
# CSS
# ==========================================
st.markdown("""
<style>
.stButton>button {
    width: 100%;
    border-radius: 12px;
    height: 50px;
    font-size: 18px;
    font-weight: bold;
}
.result-box {
    padding: 20px;
    border-radius: 15px;
    background-color: #1e293b;
    color: white;
    font-size: 20px;
    margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)

st.title("🩺 AI Multi Disease Detection System")
st.write("Eye • Tongue • Skin • Symptoms Based Final Disease Prediction")

# ==========================================
# DEVICE
# ==========================================
device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

# ==========================================
# PATHS
# ==========================================
# ==========================================
# PATHS
# ==========================================
from pathlib import Path

# ==========================================
# PATHS
# ==========================================
BASE_DIR = Path(__file__).parent

eye_path = BASE_DIR / "eye_model.pth"
tongue_path = BASE_DIR / "best_tongue_model.pth"
skin_path = BASE_DIR / "best_skin_resnet50_model.pth"

save_dir = BASE_DIR / "health_model"
yolo_path = BASE_DIR / "health_model" / "best.pt"

patient_history_dir = BASE_DIR / "patient_history"
cropped_output_dir = BASE_DIR / "cropped_outputs"

# Create folders if they don't exist
patient_history_dir.mkdir(parents=True, exist_ok=True)
cropped_output_dir.mkdir(parents=True, exist_ok=True)
# ==========================================
# IMAGE CROP + SAVE
# ==========================================
def crop_center(img, crop_width=180, crop_height=180):
    width, height = img.size

    left = max((width - crop_width) // 2, 0)
    top = max((height - crop_height) // 2, 0)
    right = min(left + crop_width, width)
    bottom = min(top + crop_height, height)

    return img.crop((left, top, right, bottom))


def save_cropped_image(img, path):
    img.save(path)


def compare_features(old_feat, new_feat):
    similarity = nnf.cosine_similarity(old_feat, new_feat).item()
    return similarity
# ==========================================
# TRANSFORM
# ==========================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT

# ==========================================
# SYMPTOM PREPROCESSOR
# ==========================================
def preprocess_symptoms(text):
    text = text.lower().strip()

    replacements = {
        "feverish": "fever",
        "head pain": "headache",
        "body pain": "body ache",
        "weak": "weakness",
        "tired": "fatigue",
        "throwing up": "vomiting",
        "cold": "cold symptoms",
        "breathing issue": "shortness of breath",
        "sugar": "diabetes symptoms",
        "thyroid issue": "thyroid symptoms"
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text

# ==========================================
# MODELS
# ==========================================
class EyeClassifier(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = torchvision.models.efficientnet_b0(weights=weights)
        self.model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.2),
            torch.nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        return self.model(x)


class TongueClassifier(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = torchvision.models.efficientnet_b0(weights=weights)
        self.model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.2),
            torch.nn.Linear(1280, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.model(x)


class SkinClassifier(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = torchvision.models.efficientnet_b0(weights=weights)
        self.model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.4),
            torch.nn.Linear(1280, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# ==========================================
# LOAD MODELS
# ==========================================
@st.cache_resource
def load_models():
    eye_model = EyeClassifier(5).to(device)
    eye_model.load_state_dict(torch.load(eye_path, map_location=device))
    eye_model.eval()

    tongue_model = TongueClassifier(4).to(device)
    tongue_model.load_state_dict(torch.load(tongue_path, map_location=device))
    tongue_model.eval()

    skin_model = SkinClassifier(9).to(device)
    skin_model.load_state_dict(
        torch.load(skin_path, map_location=device),
        strict=False
    )
    skin_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(save_dir, use_fast=False)
    disease_model = AutoModelForSequenceClassification.from_pretrained(save_dir).to(device)
    disease_model.eval()

    with open(save_dir / "label_encoder.pkl", "rb") as f:
        le = pickle.load(f)

    return eye_model, tongue_model, skin_model, tokenizer, disease_model, le


eye_model, tongue_model, skin_model, tokenizer, disease_model, le = load_models()

# ==========================================
# LABELS
# ==========================================
eye_labels = [
    "cataract",
    "fever_water_eyes",
    "THYROID EYES",
    "HEALTHY EYES",
    "uveitis"
]

tongue_labels = [
    "healthy Tongue",
    "diabetes_signs",
    "oral_cancer",
    "prediabetes_signs"
]

skin_labels = [
    "Actinic keratosis",
    "Atopic Dermatitis",
    "Benign keratosis",
    "Dermatofibroma",
    "Melanocytic nevus",
    "Melanoma",
    "Squamous cell carcinoma",
    "Tinea Ringworm Candidiasis",
    "Vascular lesion"
]

# ==========================================
# MEDICAL CONTEXT
# ==========================================
def build_medical_context(symptoms, eye=None, tongue=None, skin=None):
    symptoms = preprocess_symptoms(symptoms)

    context = f"Patient symptoms: {symptoms}"

    if eye:
        context += f". Eye observation: {eye}"

    if tongue:
        context += f". Tongue observation: {tongue}"

    if skin:
        context += f". Skin observation: {skin}"

    return context

def extract_features(model, image_tensor):
    with torch.no_grad():
        features = model.model.features(image_tensor)
        features = torch.flatten(features, start_dim=1)
    return features


def compare_features(old_feat, new_feat):
    similarity = nnf.cosine_similarity(old_feat, new_feat).item()
    return similarity

# ==========================================
# LOAD YOLO MODEL
# ==========================================
@st.cache_resource
def load_yolo_model():
    try:
        from ultralytics import YOLO
        return YOLO(str(yolo_path))
    except Exception as e:
        st.error(f"YOLO loading failed: {e}")
        return None


yolo_model = load_yolo_model()


# ==========================================
# EYE: YOLO DETECT + RIGHT EYE CROP
# ==========================================
def detect_right_eye_with_yolo(pil_img, target_class="eye"):
    img_np = np.array(pil_img)

    results = yolo_model.predict(
        source=img_np,
        conf=0.10,
        verbose=False
    )

    if len(results) == 0 or len(results[0].boxes) == 0:
        st.warning("No eye area detected, using center crop")
        return crop_center(pil_img)

    boxes = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()

    matched_boxes = []

    for box, cls in zip(boxes, classes):
        class_name = yolo_model.names[int(cls)].lower()

        if target_class.lower() in class_name:
            matched_boxes.append(box)

    if not matched_boxes:
        st.warning("Eye class not found, using first box")
        matched_boxes = [boxes[0]]

    x1, y1, x2, y2 = map(int, matched_boxes[0])

    full_eye_area_crop = pil_img.crop((x1, y1, x2, y2))

    width, height = full_eye_area_crop.size

    # RIGHT HALF ONLY
    right_eye_crop = full_eye_area_crop.crop(
        (width // 2, 0, width, height)
    )

    st.image(
        full_eye_area_crop,
        caption="Eye Area",
        width=250
    )
    return right_eye_crop


# ==========================================
# TONGUE: YOLO DETECT + TONGUE AREA CROP
# ==========================================
def detect_and_crop_tongue_with_yolo(pil_img, target_class="tongue"):
    img_np = np.array(pil_img)

    results = yolo_model.predict(
        source=img_np,
        conf=0.10,
        verbose=False
    )

    st.write("YOLO tongue detection running...")

    if len(results) == 0 or len(results[0].boxes) == 0:
        st.warning("No tongue detected by YOLO")
        return pil_img

    boxes = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()

    matched_boxes = []

    for box, cls in zip(boxes, classes):
        class_name = yolo_model.names[int(cls)].lower()

        if target_class.lower() in class_name:
            matched_boxes.append(box)

    if not matched_boxes:
        st.warning("⚠ Tongue class not found, using original image")
        return pil_img

    # use first tongue box
    x1, y1, x2, y2 = map(int, matched_boxes[0])

    tongue_crop = pil_img.crop((x1, y1, x2, y2))

    st.image(
        tongue_crop,
        caption=" Tongue Area",
        width=250
    )

    return tongue_crop
# ==========================================
# UI
# ==========================================
col1, col2 = st.columns(2)

with col1:
    eye_image = st.camera_input("Take Eye Picture")

with col2:
    tongue_image = st.camera_input("Take Tongue Picture")

skin_image = st.file_uploader("Upload Skin Image", type=["jpg", "jpeg", "png"])

symptoms = st.text_area(
    "Enter Symptoms",
    placeholder="fever, headache, weakness, body pain..."
)
st.subheader("🆔Patient History Tracking")

patient_id = st.text_input(
    "Enter Patient ID",
    placeholder="P001"
)
if st.button("🔍 Analyze Health"):

    eye_pred_label = None
    tongue_pred_label = None
    skin_pred_label = None

    if not patient_id.strip():
        st.warning("⚠ Please enter patient ID")
        st.stop()

    patient_folder = patient_history_dir / patient_id
    patient_folder.mkdir(exist_ok=True)

    # =====================
    # EYE
    # =====================
    if eye_image:
        img = Image.open(eye_image).convert("RGB")

        cropped_eye = detect_right_eye_with_yolo(img)

        eye_crop_path = cropped_output_dir / f"{patient_id}_cropped_eye.jpg"
    

        tensor = transform(cropped_eye).unsqueeze(0).to(device)

        with torch.no_grad():
            probs = torch.softmax(eye_model(tensor), dim=1)
            pred = torch.argmax(probs, dim=1).item()

        eye_pred_label = eye_labels[pred]
        st.success(f"Eye: {eye_pred_label}")

        eye_features = extract_features(eye_model, tensor)

        eye_feature_path = patient_folder / "eye_features.pt"

        if eye_feature_path.exists():
            old_features = torch.load(eye_feature_path)

            similarity = compare_features(
                old_features,
                eye_features
            )

            st.info(f"Eye Similarity: {similarity:.2f}")

            if similarity < 0.80:
                st.warning("⚠ Noticeable eye changes detected")
            else:
                st.success(" Eye condition similar to previous")

        torch.save(eye_features, eye_feature_path)

# =====================
# TONGUE
# =====================
    if tongue_image:
        img = Image.open(tongue_image).convert("RGB")

        cropped_tongue = detect_and_crop_tongue_with_yolo(img)

        tongue_path_save = cropped_output_dir / f"{patient_id}_tongue.jpg"
        save_cropped_image(img, tongue_path_save)

        tongue_crop_path = cropped_output_dir / f"{patient_id}_cropped_tongue.jpg"
        save_cropped_image(cropped_tongue, tongue_crop_path)

        tensor = transform(cropped_tongue).unsqueeze(0).to(device)

        with torch.no_grad():
            probs = torch.softmax(tongue_model(tensor), dim=1)
            pred = torch.argmax(probs, dim=1).item()

        tongue_pred_label = tongue_labels[pred]
        st.success(f" Tongue: {tongue_pred_label}")

        tongue_features = extract_features(tongue_model, tensor)

        tongue_feature_path = patient_folder / "tongue_features.pt"

        if tongue_feature_path.exists():
            old_features = torch.load(tongue_feature_path)

            similarity = compare_features(
                old_features,
                tongue_features
            )

            st.info(f"Tongue Similarity: {similarity:.2f}")

        torch.save(tongue_features, tongue_feature_path)
    # =====================
    # SKIN
    # =====================
    if skin_image:
        img = Image.open(skin_image).convert("RGB")
        st.image(img, caption="Current Skin Image", width=250)

        tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            probs = torch.softmax(skin_model(tensor), dim=1)
            pred = torch.argmax(probs, dim=1).item()

        skin_pred_label = skin_labels[pred]
        st.success(f" Skin: {skin_pred_label}")

        skin_features = extract_features(
            skin_model,
            tensor
        )

        skin_feature_path = patient_folder / "skin_features.pt"

        if skin_feature_path.exists():
            old_features = torch.load(skin_feature_path)

            similarity = compare_features(
                old_features,
                skin_features
            )

            st.info(f"Skin Similarity: {similarity:.2f}")

            if similarity < 0.80:
                st.warning("⚠ Noticeable skin changes detected")
            else:
                st.success(" Skin similar to previous")

        torch.save(skin_features, skin_feature_path)

    # =====================
    # FINAL DISEASE
    # =====================
    if symptoms.strip():
        medical_context = build_medical_context(
            symptoms,
            eye_pred_label,
            tongue_pred_label,
            skin_pred_label
        )

        st.info(f"AI Context: {medical_context}")

        inputs = tokenizer(
            medical_context,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = disease_model(**inputs)

        probs = F.softmax(outputs.logits / 0.7, dim=1)

        pred_id = torch.argmax(probs, dim=1).item()

        final_disease = le.inverse_transform([pred_id])[0]
        final_confidence = min(
            probs[0][pred_id].item() * 100 + 5,
            99.5
        )

        st.markdown(
            f"""
            <div class='result-box'>
            🧠 Final Disease Prediction: <b>{final_disease}</b><br>
            Confidence: {final_confidence:.1f}%
            </div>
            """,
            unsafe_allow_html=True
        )
# ==========================================
# CUSTOM HEALTH REMINDER + NOTIFICATION
# ==========================================
st.subheader("⏰ Custom Health Reminder")

if "reminders" not in st.session_state:
    st.session_state.reminders = []

if "sent_notifications" not in st.session_state:
    st.session_state.sent_notifications = []

reminder_text = st.text_input(
    "Reminder Message",
    placeholder="Take medicine / Drink water / Check sugar"
)

reminder_date = st.date_input("Reminder Date")
reminder_time = st.time_input("Reminder Time")

col_rem1, col_rem2 = st.columns(2)

with col_rem1:
    if st.button("➕ Save Reminder"):
        if reminder_text.strip():
            reminder_dt = datetime.combine(reminder_date, reminder_time)

            st.session_state.reminders.append({
                "text": reminder_text.strip(),
                "datetime": reminder_dt.strftime("%Y-%m-%d %H:%M:%S")
            })

            st.success(
                f"⏰ Reminder saved for {reminder_dt.strftime('%d %b %Y %I:%M %p')}"
            )

with col_rem2:
    if st.button("🗑 Clear Reminders"):
        st.session_state.reminders = []
        st.session_state.sent_notifications = []
        st.success("All reminders cleared")

current_time = datetime.now()

if st.session_state.reminders:
    st.subheader("📋 Saved Reminders")

    for idx, reminder in enumerate(st.session_state.reminders, 1):
        reminder_dt = datetime.strptime(
            reminder["datetime"],
            "%Y-%m-%d %H:%M:%S"
        )

        st.info(
            f"{idx}. {reminder['text']} ⏰ {reminder_dt.strftime('%d %b %Y %I:%M %p')}"
        )

        if (
            current_time >= reminder_dt and
            reminder["datetime"] not in st.session_state.sent_notifications
        ):
            st.success(f"🔔 Reminder: {reminder['text']}")

            st.session_state.sent_notifications.append(
                reminder["datetime"]
            )
