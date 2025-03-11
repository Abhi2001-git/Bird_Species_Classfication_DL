import streamlit as st
import tensorflow
import keras
from keras.saving import load_model
import numpy as np
from PIL import Image

# Load the trained deep learning model
MODEL_PATH = r"my_best_cnn_vgg16.h5"
model = load_model(MODEL_PATH)

# Class labels (6 bird species)
class_labels = [
    'AMERICAN GOLDFINCH', 'BARN OWL', 'CARMINE BEE-EATER',
    'DOWNY WOODPECKER', 'EMPEROR PENGUIN', 'FLAMINGO'
]

# Streamlit UI Configuration
st.set_page_config(page_title="Bird Species Classifier", page_icon="🦜", layout="centered")

# App Title
st.markdown(
    "<h1 style='text-align: center; color: #FF5733;'>🦜 Bird Species Classifier 🦜</h1>", 
    unsafe_allow_html=True
)

# Model Description
st.markdown(
    """
    <p style='text-align: center; color: #3498db; font-size: 20px;'>
        This deep learning model can classify bird images into one of the following six species:
        AMERICAN GOLDFINCH, BARN OWL, CARMINE BEE-EATER, 
        DOWNY WOODPECKER, EMPEROR PENGUIN, and FLAMINGO.
    </p>
    <p style='text-align: center; color: #e74c3c; font-size: 20px;font-weight: bold;'>
        Upload an image and let AI predict the species!
    </p>
    """, 
    unsafe_allow_html=True
)


# File uploader
uploaded_file = st.file_uploader("📤 **Upload a bird image** (JPG, PNG, JPEG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="📸 Uploaded Image", use_container_width=True)

    # Preprocess image
    image = image.resize((224, 224))  # Resize to model input size
    image = np.array(image) / 255.0   # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Prediction
    with st.spinner("🔍 Analyzing the image..."):
        prediction = model.predict(image)

    # Get the highest probability class
    class_index = np.argmax(prediction)
    predicted_label = class_labels[class_index]
    confidence = np.max(prediction) * 100  # Convert to percentage

    # Display Prediction Result
    st.success(f"🎯 **Predicted Bird Species:** {predicted_label}")
    st.progress(int(confidence))  # Confidence Bar
    st.write(f"📊 **Confidence Score:** {confidence:.2f}%")

    # Bird fun facts (Optional)
    bird_facts = {
        'AMERICAN GOLDFINCH': "🌻 Loves sunflower seeds and has a bright yellow plumage.",
        'BARN OWL': "🦉 Known for its heart-shaped face and silent flight.",
        'CARMINE BEE-EATER': "🐝 Feeds on bees and has stunning crimson feathers.",
        'DOWNY WOODPECKER': "🔨 Small but mighty, often seen drumming on trees.",
        'EMPEROR PENGUIN': "❄️ The largest penguin species, thriving in Antarctica.",
        'FLAMINGO': "🦩 Gets its pink color from its shrimp-based diet!"
    }
    
    st.info(bird_facts.get(predicted_label, "🐦 A fascinating bird species!"))

# Footer with Social Links
st.markdown(
    """
    <hr>
    <h4 style="text-align:center;">Developed by <b>Abhisikta Moharana</b></h4>
    <p style="text-align:center;">
        <a href="https://www.linkedin.com/in/abhisikta-moharana-983052270" target="_blank" style="text-decoration:none; color:#d63384;">
        🌐 LinkedIn
        </a> |
        <a href="https://github.com/Abhi2001-git" target="_blank" style="text-decoration:none; color:#d63384;">
        🖥️ GitHub
        </a> |
        <a href="mailto:abhisikta.moharana2001@gmail.com" style="text-decoration:none; color:#d63384;">
        📩 Email
        </a>
    </p>
    """,
    unsafe_allow_html=True
)
