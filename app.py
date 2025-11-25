import os
import streamlit as st
import re
from datetime import datetime
import pandas as pd
from io import BytesIO
import time 
import json # New import for parsing the model response

# Removed: from dotenv import load_dotenv # 💥 Removed for st.secrets

# --- Configuration ---
st.set_page_config(
    page_title="🪪 Myanmar Driving License Extractor (AI OCR)",
    layout="wide"
)

from google import genai
from google.genai import types
from PIL import Image

# Initialize the Gemini Client
try:
    # 💥 CHANGE: Use st.secrets to securely load the API key
    api_key = st.secrets["GEMINI_API_KEY"] 
    client = genai.Client(api_key=api_key) # Pass the key explicitly
except KeyError:
    # Handle the specific error if the key is missing in the Streamlit secrets
    st.error("Error: GEMINI_API_KEY not found in Streamlit Secrets. Please configure your secrets file/settings.")
    st.stop()
except Exception as e:
    # Original error for general initialization issues
    st.error(f"Error initializing AI client. Please ensure your API key is valid. Details: {e}")
    st.stop()


# --- 2. Data Extraction Prompt and Schema (Updated for Burmese and Confidence) ---

# Define the expected output structure
extraction_schema = {
    "type": "object",
    "properties": {
        # English/Latin Script Fields
        "license_no": {"type": "string", "description": "The driving license number, typically like 'A/123456/22'."},
        "name": {"type": "string", "description": "The full name of the license holder in Latin script."},
        "nrc_no": {"type": "string", "description": "The NRC ID number, typically like '12/MASANA(N)123456'."},
        "date_of_birth": {"type": "string", "description": "The date of birth in DD-MM-YYYY format."},
        "blood_type": {"type": "string", "description": "The blood type, e.g., 'A+', 'B', 'O-', 'AB'."},
        "valid_up": {"type": "string", "description": "The license expiry date in DD-MM-YYYY format."},
        
        # Burmese/Myanmar Script Fields (Added for cross-validation and display)
        "name_myanmar": {"type": "string", "description": "The full name of the license holder in Myanmar script (အမည်)."},
        "date_of_birth_myanmar": {"type": "string", "description": "The date of birth in Myanmar script (မွေးသကရာဇ်)."},
        "valid_up_myanmar": {"type": "string", "description": "The license expiry date in Myanmar script (ကုန်ဆုံးရက်)."},
        
        # Confidence Score (Proxy for extraction quality)
        "extraction_confidence": {"type": "number", "description": "The model's self-assessed confidence score for the entire extraction, from 0.0 (low) to 1.0 (high)."}
    },
    "required": [
        "license_no", "name", "nrc_no", "date_of_birth", "blood_type", "valid_up",
        "name_myanmar", "date_of_birth_myanmar", "valid_up_myanmar", "extraction_confidence"
    ]
}

# The main prompt for the model (Updated for Burmese and Confidence)
EXTRACTION_PROMPT = """
Analyze the provided image, which is a Myanmar Driving License.
Extract ALL data fields, including both the Latin script (English) and Myanmar script (Burmese) values, and return the result strictly as a JSON object matching the provided schema.

The Burmese fields to extract are:
- အမှတ် (License No) -> Use the Latin script for 'license_no'
- အမည် (Name) -> Extract in Myanmar script for 'name_myanmar'
- မှတ်ပုံတင်အမှတ် (NRC No) -> Use the Latin script for 'nrc_no'
- မွေးသကရာဇ် (Date of Birth) -> Extract in Myanmar script for 'date_of_birth_myanmar'
- သွေးအုပ်စု (Blood Type) -> Use the Latin script for 'blood_type'
- ကုန်ဆုံးရက် (Valid Up/Expiry Date) -> Extract in Myanmar script for 'valid_up_myanmar'

Ensure all Latin dates are in the DD-MM-YYYY format.
Finally, provide your best self-assessed confidence for the entire extraction on a scale of 0.0 to 1.0 for 'extraction_confidence'.
If a field is not found, return an empty string "" for that value.
Do not include any extra text or formatting outside of the JSON object.
"""

# --- 3. File Handling Function (Only PIL remains) ---

def handle_file_to_pil(uploaded_file):
    """Converts uploaded file or bytes to a PIL Image object."""
    if uploaded_file is None:
        return None
        
    file_bytes = uploaded_file.read() if hasattr(uploaded_file, 'read') else uploaded_file
    try:
        # Use PIL to open directly from bytes
        image_pil = Image.open(BytesIO(file_bytes))
        return image_pil
    except Exception as e:
        st.error(f"Error converting file to image: {e}")
        return None
        
# --- 4. AI Extraction Logic ---

def run_structured_extraction(image_pil):
    """
    Uses the AI API to analyze the image and extract structured data.
    """
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[EXTRACTION_PROMPT, image_pil],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=extraction_schema,
                temperature=0.0, # Use low temperature for deterministic data extraction
            )
        )
        
        # The response.text is a JSON string matching the schema
        structured_data = json.loads(response.text)
        return structured_data
        
    except genai.errors.APIError as e:
        # Changed output text
        st.error(f"AI API Error: Could not process the image. Details: {e}")
        return None
    except Exception as e:
        # Changed output text
        st.error(f"An unexpected error occurred during AI processing: {e}")
        return None

# --- 5. Helper Functions (Updated to include Burmese fields and Confidence) ---

def create_downloadable_files(extracted_dict):
    """Formats the extracted data into CSV, TXT, and DOC formats."""
    
    # 1. Prepare display dictionary with both languages and confidence
    results_dict = {
        "License No (A/123.../22)": extracted_dict.get('license_no', ''),
        "Name (English)": extracted_dict.get('name', ''),
        "အမည် (Myanmar)": extracted_dict.get('name_myanmar', ''),
        "NRC No (12/...)": extracted_dict.get('nrc_no', ''),
        "Date of Birth (DD-MM-YYYY)": extracted_dict.get('date_of_birth', ''),
        "မွေးသကရာဇ် (Myanmar)": extracted_dict.get('date_of_birth_myanmar', ''),
        "Blood Type": extracted_dict.get('blood_type', ''),
        "Valid Up (DD-MM-YYYY)": extracted_dict.get('valid_up', ''),
        "ကုန်ဆုံးရက် (Myanmar)": extracted_dict.get('valid_up_myanmar', ''),
        "Extraction Confidence (0.0 - 1.0)": f"{extracted_dict.get('extraction_confidence', 0.0):.2f}"
    }
    
    # 2. Prepare TXT content
    txt_content = "\n".join([f"{key}: {value}" for key, value in results_dict.items()])
    
    # 3. Prepare DataFrame for CSV
    df = pd.DataFrame(results_dict.items(), columns=['Field', 'Value'])
    
    csv_buffer = BytesIO()
    df.to_csv(csv_buffer, index=False, encoding='utf-8') # Ensure UTF-8 for Burmese characters
    csv_content = csv_buffer.getvalue()
    
    # 4. Prepare DOC content (tab-separated for easy copy-paste)
    doc_content = "\n".join([f"{key}\t{value}" for key, value in results_dict.items()])
    
    return txt_content, csv_content, doc_content, results_dict


# --- 6. UI and Execution Flow (Updated) ---

def process_image_and_display(original_image_pil, unique_key_suffix):
    """
    Performs AI extraction and displays results. 
    """
    st.subheader("Processing Image...")
    
    with st.spinner("Running AI Structured Extraction (English & Myanmar OCR)..."):
        time.sleep(1) 
        
        # 1. Run Structured Extraction
        raw_extracted_data = run_structured_extraction(original_image_pil)
        
        if raw_extracted_data is None:
             st.stop() 

        # 2. Prepare data for display/download
        txt_file, csv_file, doc_file, extracted_data = create_downloadable_files(raw_extracted_data)
        
    st.success(f"Extraction Complete! Confidence: **{extracted_data['Extraction Confidence (0.0 - 1.0)']}**")
        
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Uploaded Image")
        # Display the original PIL image directly
        st.image(original_image_pil, use_column_width=True) # Changed to use_column_width for better fit
        
    with col2:
        st.header("Extraction Results")
        
        # --- Results Form (Updated with Burmese fields) ---
        form_key = f"results_form_{unique_key_suffix}"
        with st.form(form_key): 
            st.text_input("License No", value=extracted_data["License No (A/123.../22)"])
            st.text_input("Name (English)", value=extracted_data["Name (English)"])
            st.text_input("အမည် (Myanmar)", value=extracted_data["အမည် (Myanmar)"])
            st.text_input("NRC No", value=extracted_data["NRC No (12/...)"])
            st.text_input("Date of Birth (Eng)", value=extracted_data["Date of Birth (DD-MM-YYYY)"])
            st.text_input("မွေးသကရာဇ် (Myan)", value=extracted_data["မွေးသကရာဇ် (Myanmar)"])
            st.text_input("Blood Type", value=extracted_data["Blood Type"])
            st.text_input("Valid Up (Eng)", value=extracted_data["Valid Up (DD-MM-YYYY)"])
            st.text_input("ကုန်ဆုံးရက် (Myan)", value=extracted_data["ကုန်ဆုံးရက် (Myanmar)"])
            st.text_input("Confidence Score", value=extracted_data["Extraction Confidence (0.0 - 1.0)"])
            st.form_submit_button("Acknowledge & Validate") 
            
        st.subheader("Download Data")
        
        # --- Download Buttons ---
        st.download_button(
            label="⬇️ Download CSV", 
            data=csv_file, 
            file_name=f"license_data_{unique_key_suffix}.csv", 
            mime="text/csv", 
            key=f"download_csv_{unique_key_suffix}"
        )
        st.download_button(
            label="⬇️ Download Plain Text", 
            data=txt_file, 
            file_name=f"license_data_{unique_key_suffix}.txt", 
            mime="text/plain", 
            key=f"download_txt_{unique_key_suffix}" 
        )
        st.download_button(
            label="⬇️ Download Word (.doc)", 
            data=doc_file, 
            file_name=f"license_data_{unique_key_suffix}.doc", 
            mime="application/msword", 
            key=f"download_doc_{unique_key_suffix}" 
        )

# --- Main App Body ---

st.title("🪪 Myanmar License Extractor (AI OCR)")
st.caption("Now supports **Myanmar (Burmese)** script extraction and provides an **AI Confidence Score** for the results.")

# --- Tab Setup ---
tab1, tab2 = st.tabs(["📷 Live Capture (Scanner)", "⬆️ Upload File"])

current_time_suffix = str(time.time()).replace('.', '') 

# --- Live Capture Tab ---
with tab1:
    st.header("Live Document Capture")
    st.write("Use your device's camera to scan the front of the driving license.")
    captured_file = st.camera_input("Place the license clearly in the frame and click 'Take Photo'", key="camera_input")
    
    if captured_file is not None:
        image_pil = handle_file_to_pil(captured_file)
        
        if image_pil is not None:
            process_image_and_display(
                image_pil, 
                f"live_{current_time_suffix}"
            )
        else:
            st.error("Could not read the captured image data. Please ensure the camera capture was successful.")

# --- Upload File Tab ---
with tab2:
    st.header("Upload Image File")
    st.write("Upload a clear photo or scan of the front of the driving license.")
    uploaded_file = st.file_uploader("Upload License Image", type=['jpg', 'png', 'jpeg'], key="file_uploader")
    
    if uploaded_file is not None:
        image_pil = handle_file_to_pil(uploaded_file)
        
        if image_pil is not None:
            process_image_and_display(
                image_pil, 
                f"upload_{current_time_suffix}"
            )
        else:
            st.error("Could not read the uploaded image data. Please ensure the file is a valid image.")
