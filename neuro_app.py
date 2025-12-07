import streamlit as st
import time
import random
from PIL import Image
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# Load the model ONCE at the start
# (Cache it so we don't reload it every time someone clicks a button)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('brain_tumor_model.h5')

model = load_model()

#Section 1: The Page Setup
st.set_page_config(
    page_title="NeuroScan AI",
    page_icon="🧠",
    layout="wide"
)

# Section 2: Custom Styling (CSS)
# We inject some CSS to make it look dark and medical.
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 5px;
        height: 3em;
        width: 100%;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# Section 3: The Sidebar (Control Panel)
st.sidebar.header("⚙️ System Settings")
# These inputs don't do math, they just make the user feel in control
patient_id = st.sidebar.text_input("Patient ID", "PT-4920")
scan_type = st.sidebar.selectbox("MRI Sequence", ["T1-Weighted", "T2-Weighted", "FLAIR", "T1-Contrast", "T2-Contrast", "DWI", "SWI"])
st.sidebar.divider()
st.sidebar.info("System Status: ONLINE")

# Section 4: The Main Title
# columns allow us to put things side-by-side
col1, col2 = st.columns([1, 5]) 

with col1:
    # A placeholder medical icon
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966486.png", width=80)

with col2:
    st.title("NeuroScan AI: Tumor Detection")
    st.write("Advanced Neural Network for MRI Analysis")

st.divider()

# Section 5: The Logic Function (The "Brain")
def analyze_mri(image):
    # 1. Preprocess the image to match the Training (150x150)
    size = (150, 150)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image = image.convert('RGB')
    img_array = np.asarray(image)
    img_array = img_array / 255.0  # Normalize (0-1)
    img_reshape = np.expand_dims(img_array, axis=0) # Add batch dimension
    
    # 2. Predict
    prediction = model.predict(img_reshape)
    
    # 3. Interpret Result
    # The output is a number between 0 and 1
    # 0 = No Tumor, 1 = Tumor
    score = prediction[0][0]
    
    # --- NEW GUARDRAIL LOGIC ---
    # Convert raw score to a "Confidence" percentage (0-100)
    # If score is > 0.5, it leans Tumor. If < 0.5, it leans Normal.
    if score > 0.5:
        label = "TUMOR"
        confidence = score
    else:
        label = "NORMAL"
        confidence = 1.0 - score
        
    # THE GATEKEEPER:
    # If the AI is confused (confidence is between 50% and 80%), reject it.
    if confidence < 0.80:
        return "UNKNOWN", round(confidence * 100, 2)
        
    # If we pass the gate, return the real result
    return label, round(confidence * 100, 2)


    if score > 0.5:
        has_tumor = True
        confidence = round(score * 100, 2)
    else:
        has_tumor = False
        confidence = round((1 - score) * 100, 2)
        
    return has_tumor, confidence

# Section 6: The User Interface
uploaded_file = st.file_uploader("Upload Brain MRI Scan...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Divide screen into two columns: Left (Image), Right (Results)
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("Patient Scan")
        # 'use_column_width=True' makes the image fit the box perfectly
        image = Image.open(uploaded_file)
        st.image(image, caption=f"ID: {patient_id} | {scan_type}", use_column_width=True)

    with col_b:
        st.subheader("Diagnostic Analysis")
        # Update the button click section:
        if st.button("RUN AI DIAGNOSIS"):
            with st.spinner("Processing Neural Layers..."):
                result_label, confidence = analyze_mri(image)
            
            # --- NEW UI LOGIC ---
            if result_label == "UNKNOWN":
                st.warning(f"⚠️ IMAGE UNRECOGNIZED (Low Confidence: {confidence}%)")
                st.write("The AI is not sure this is a brain MRI. Please upload a clear medical scan.")
                
            elif result_label == "TUMOR":
                st.error("⚠️ ABNORMALITY DETECTED")
                st.metric(label="Confidence Score", value=f"{confidence}%", delta="High Risk")
                st.progress(float(confidence) / 100)
                st.write("Target area identified. Neurosurgery consult recommended.")
                
            elif result_label == "NORMAL":
                st.success("✅ SCAN NORMAL")
                st.metric(label="Confidence Score", value=f"{confidence}%")
                st.write("No pathological anomalies detected.")

else:
    # What to show when nothing is uploaded yet
    st.info("Please upload a DICOM or JPEG image to begin analysis.")