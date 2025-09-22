import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Set the title and icon of the Streamlit app
st.set_page_config(page_title="Pneumonia Detection from X-Rays", page_icon="🫁")
st.title("🫁 Pneumonia Detection from Chest X-Rays")
st.write("Upload a chest X-ray image, and the model will predict whether it shows signs of Pneumonia.")

# --- Load the Trained Model ---
@st.cache_resource
def load_keras_model():
    """Load the Keras model and re-compile it."""
    try:
        # FIX 1: The function must be called through tf.keras.models
        model = tf.keras.models.load_model('chest_xray_model.h5', compile=False)
        model.compile(optimizer='adam', 
                      loss='binary_crossentropy', 
                      metrics=['accuracy'])
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

model = load_keras_model()

# --- Image Preprocessing Function ---
def preprocess_image(img):
    """Preprocesses the uploaded image to fit the model's input requirements."""
    # FIX: Add this line to ensure the image is in RGB format (3 channels)
    img = img.convert('RGB')
    
    img = img.resize((150, 150))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# --- File Uploader and Prediction Logic ---
uploaded_file = st.file_uploader("Choose a chest X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded X-Ray.', use_container_width=True)
    
    if st.button('Analyze Image'):
        if model is not None:
            with st.spinner('Model is analyzing the image...'):
                processed_img = preprocess_image(img)
                prediction = model.predict(processed_img)
                
                st.subheader("Prediction Result:")
                
                if prediction[0][0] > 0.5:
                    confidence = prediction[0][0] * 100
                    st.error(f"**Result: Pneumonia** ({confidence:.2f}% confidence)")
                else:
                    confidence = (1 - prediction[0][0]) * 100
                    st.success(f"**Result: Normal** ({confidence:.2f}% confidence)")
        else:
            st.error("The model could not be loaded. Please check the terminal for errors.")