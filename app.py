import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from fpdf import FPDF
from datetime import datetime
import os
import re
from transformers import pipeline # New Hugging Face import

# --- Page Configuration ---
st.set_page_config(page_title="Pneumonia Detection from X-Rays", page_icon="🫁")
st.title("🫁 Pneumonia Detection from Chest X-Rays")
st.write("Upload a chest X-ray image, and the model will predict whether it shows signs of Pneumonia.")

# --- Model Loading ---
@st.cache_resource
def load_keras_model():
    """Load the Keras model and re-compile it."""
    try:
        model = tf.keras.models.load_model('chest_xray_model.h5', compile=False)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

# --- NEW: Hugging Face Model Loading ---
@st.cache_resource
def load_hf_pipeline():
    """Loads the Hugging Face text generation pipeline."""
    try:
        # We use a small, instruction-tuned model for this task
        generator = pipeline('text2text-generation', model='google/flan-t5-small')
        return generator
    except Exception as e:
        st.error(f"Error loading the Hugging Face model: {e}")
        return None

model = load_keras_model()
hf_generator = load_hf_pipeline()

# --- Image Preprocessing ---
def preprocess_image(img):
    """Preprocesses the uploaded image to fit the model's input requirements."""
    img = img.convert('RGB')
    img = img.resize((150, 150))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# --- NEW: Hugging Face Report Generation Function ---
@st.cache_data
def generate_hf_report_text(_generator): # The underscore tells Streamlit to ignore the generator object for hashing
    """Calls the local Hugging Face model to generate informational text."""
    if _generator is None:
        return "Hugging Face model not loaded. Additional details unavailable."
    
    try:
        # This prompt is tailored for an instruction-following model like Flan-T5
        prompt = """
        A chest X-ray analysis suggests "Pneumonia". Write a brief, informative report with three sections: 
        1. Possible General Causes.
        2. General Suggestions for Recovery.
        3. Types of Doctors to Consult.
        
        Write the information in clear, narrative paragraphs and use headings for each section. Start with a disclaimer that this is not medical advice.
        """
        outputs = _generator(prompt, max_length=256, clean_up_tokenization_spaces=True)
        return outputs[0]['generated_text']
    except Exception as e:
        return f"Could not generate additional details due to an error: {e}"

# --- PDF Report Generation Function (Unchanged) ---
def create_pdf_report(image, diagnosis, confidence, additional_text=""):
    # This function remains the same as before
    pdf = FPDF()
    pdf.add_page()
    effective_width = pdf.w - 2 * pdf.l_margin
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(effective_width, 10, 'Chest X-Ray Analysis Report', ln=True, align='C')
    pdf.ln(10)
    pdf.set_font('Arial', '', 12)
    pdf.cell(effective_width, 10, f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(5)
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(effective_width, 10, 'Prediction Result', ln=True)
    pdf.set_font('Arial', '', 12)
    pdf.cell(effective_width, 8, f" - Diagnosis: {diagnosis}", ln=True)
    pdf.cell(effective_width, 8, f" - Confidence: {confidence:.2f}%", ln=True)
    pdf.ln(10)
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(effective_width, 10, 'Analyzed Image', ln=True)
    temp_image_path = "temp_uploaded_image.png"
    image.save(temp_image_path)
    pdf.image(temp_image_path, x=(pdf.w - 100) / 2, y=pdf.get_y(), w=100)
    os.remove(temp_image_path)
    pdf.ln(70)
    if additional_text:
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(effective_width, 10, 'Additional Information', ln=True)
        pdf.ln(2)
        pdf.set_font('Arial', '', 10)
        pdf.multi_cell(effective_width, 5, additional_text) # Simplified for direct output
        pdf.ln(5)
    pdf.set_font('Arial', 'I', 8)
    pdf.cell(effective_width, 10, 'Disclaimer: This is an AI-generated analysis and not a substitute for professional medical advice.', ln=True, align='C')
    return pdf.output()

# --- Main App Logic (Updated) ---
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
                additional_text = ""
                
                if prediction[0][0] > 0.5:
                    diagnosis = "Pneumonia"
                    confidence = prediction[0][0] * 100
                    st.error(f"**Result: {diagnosis}** ({confidence:.2f}% confidence)")
                    # NEW: Generate text using the local Hugging Face model
                    with st.spinner("Generating additional information with local model..."):
                        additional_text = generate_hf_report_text(hf_generator)
                else:
                    diagnosis = "Normal"
                    confidence = (1 - prediction[0][0]) * 100
                    st.success(f"**Result: {diagnosis}** ({confidence:.2f}% confidence)")

                pdf_bytes = create_pdf_report(img, diagnosis, confidence, additional_text=additional_text)
                
                st.download_button(
                    label="Download Full Report as PDF",
                    data=bytes(pdf_bytes),
                    file_name=f"xray_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf"
                )
        else:
            st.error("The model could not be loaded. Please check the terminal for errors.")