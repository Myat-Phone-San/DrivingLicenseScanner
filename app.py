import streamlit as st
import json
from io import BytesIO
import time
import pandas as pd
from PIL import Image
from google import genai
from google.genai import types
from google.genai.errors import APIError

# --- 1. Configuration and Initialization ---
st.set_page_config(
    page_title="🪪 မြန်မာ မှတ်ပုံတင် ထုတ်ယူခြင်း (AI OCR)", # Myanmar NRC Extractor
    layout="wide"
)

# Initialize the Gemini Client
@st.cache_resource
def initialize_gemini_client():
    """Initializes and returns the Gemini client."""
    try:
        # Load API key securely from Streamlit Secrets
        api_key = st.secrets["GEMINI_API_KEY"]
        if not api_key:
             st.error("Error: GEMINI_API_KEY is empty. Please configure your Streamlit secrets.")
             st.stop()
        client = genai.Client(api_key=api_key)
        return client
    except KeyError:
        st.error("Error: GEMINI_API_KEY not found in Streamlit Secrets. Please configure your secrets file/settings.")
        st.stop()
    except Exception as e:
        st.error(f"Error initializing AI client. Details: {e}")
        st.stop()

# Initialize client globally
client = initialize_gemini_client()


# --- 2. Data Extraction Prompt and Schema ---

# Define the expected output structure for NRC (Myanmar Fields Only)
extraction_schema = {
    "type": "object",
    "properties": {
        # Burmese/Myanmar Script Fields
        "nrc_serial_myanmar": {"type": "string", "description": "အမှတ်: The card serial number in Myanmar script (e.g., '၁၀/မစန/၉၆')."},
        "issue_date_myanmar": {"type": "string", "description": "ရက်စွဲ: The issue date in Myanmar script (e.g., '၂၆-၁၀-၂၀၁၆')."},
        "name_myanmar": {"type": "string", "description": "အမည်: The full name of the NRC holder in Myanmar script."},
        "father_name_myanmar": {"type": "string", "description": "ဖခင်အမည်: The father's name in Myanmar script."},
        "nrc_no_myanmar": {"type": "string", "description": "မှတ်ပုံတင်အမှတ်: The NRC ID number fully in Myanmar script (e.g., '၉/မထလ(နိုင်)၃၂၆၄၅၈')."},
        "date_of_birth_myanmar": {"type": "string", "description": "မွေးသကရာဇ်: The date of birth in Myanmar script."},
        "nationality_religion_myanmar": {"type": "string", "description": "လူမျိုး/ဘာသာ: The Nationality/Religion in Myanmar script."},
        "height_myanmar": {"type": "string", "description": "အရပ်: The height in Myanmar script."},
        "identifying_mark_myanmar": {"type": "string", "description": "ထင်ရှားသည့်အမှတ်အသား: The identifying mark in Myanmar script."},
        # Confidence Score
        "extraction_confidence": {"type": "number", "description": "The model's self-assessed confidence score for the entire extraction, from 0.0 (low) to 1.0 (high)."}
    },
    "required": [
        "nrc_serial_myanmar", "issue_date_myanmar", "name_myanmar", "father_name_myanmar", "nrc_no_myanmar",
        "date_of_birth_myanmar", "nationality_religion_myanmar", "height_myanmar", "identifying_mark_myanmar",
        "extraction_confidence"
    ]
}

# The main prompt for the model
EXTRACTION_PROMPT = """
Analyze the provided image, which is a Myanmar National Registration Card (NRC) or a similar identity document.
Extract ALL data fields and return the result **STRICTLY in Myanmar (Burmese) script and digits**, matching the provided JSON schema.

---
CRITICAL INSTRUCTIONS:
1. Extract ALL fields directly in Myanmar script, using Myanmar digits (၀-၉).
2. For the NRC number ('မှတ်ပုံတင်အမှတ်'), ensure the entire string is transliterated into Myanmar script (e.g., '၉/မထလ(နိုင်)၃၂၆၄၅၈').
---

Finally, provide your best self-assessed confidence for the entire extraction on a scale of 0.0 to 1.0 for 'extraction_confidence'.
If a field is not found, return an empty string "" for that value.
Do not include any extra text or formatting outside of the JSON object.
"""

# --- 3. Helper Functions ---

def handle_file_to_pil(uploaded_file):
    """Converts uploaded file or bytes to a PIL Image object."""
    if uploaded_file is None:
        return None

    file_bytes = uploaded_file.read() if hasattr(uploaded_file, 'read') else uploaded_file
    try:
        image_pil = Image.open(BytesIO(file_bytes))
        return image_pil
    except Exception as e:
        st.error(f"Error converting file to image: {e}")
        return None

def run_structured_extraction(image_pil):
    """Uses the AI API to analyze the image and extract structured data."""
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[EXTRACTION_PROMPT, image_pil],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=extraction_schema,
                temperature=0.0, # Low temperature for deterministic extraction
            )
        )
        
        # The response.text is a JSON string matching the schema
        structured_data = json.loads(response.text)
        return structured_data

    except APIError as e:
        st.error(f"AI API Error: Could not process the image. Please check API key validity and network connection. Details: {e}")
        return None
    except json.JSONDecodeError:
        st.error("AI Response Error: The model did not return valid JSON. Please try again with a clearer image.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during AI processing: {e}")
        return None

def create_downloadable_files(extracted_dict):
    """Formats the extracted data into CSV, TXT, and DOC formats."""

    # 1. Prepare display dictionary (Myanmar Fields Only)
    results_dict = {
        "၁. အမှတ် (NRC Serial)": extracted_dict.get('nrc_serial_myanmar', 'N/A'),
        "၂. ရက်စွဲ (Issue Date)": extracted_dict.get('issue_date_myanmar', 'N/A'),
        "၃. အမည် (Name)": extracted_dict.get('name_myanmar', 'N/A'),
        "၄. ဖခင်အမည် (Father\'s Name)": extracted_dict.get('father_name_myanmar', 'N/A'),
        "၅. မှတ်ပုံတင်အမှတ် (NRC No)": extracted_dict.get('nrc_no_myanmar', 'N/A'),
        "၆. မွေးသကရာဇ် (Date of Birth)": extracted_dict.get('date_of_birth_myanmar', 'N/A'),
        "၇. လူမျိုး/ဘာသာ (Nationality/Religion)": extracted_dict.get('nationality_religion_myanmar', 'N/A'),
        "၈. အရပ် (Height)": extracted_dict.get('height_myanmar', 'N/A'),
        "၉. ထင်ရှားသည့်အမှတ်အသား (Identifying Mark)": extracted_dict.get('identifying_mark_myanmar', 'N/A'),
        "AI Extraction Confidence (0.0 - 1.0)": f"{extracted_dict.get('extraction_confidence', 0.0):.2f}"
    }

    # 2. Prepare TXT content
    txt_content = "\n".join([f"{key}: {value}" for key, value in results_dict.items()])

    # 3. Prepare DataFrame for CSV
    # Only include the Burmese fields for a cleaner CSV output
    df = pd.DataFrame([
        {"Field": "အမှတ်", "Value": results_dict["၁. အမှတ် (NRC Serial)"]},
        {"Field": "ရက်စွဲ", "Value": results_dict["၂. ရက်စွဲ (Issue Date)"]},
        {"Field": "အမည်", "Value": results_dict["၃. အမည် (Name)"]},
        {"Field": "ဖခင်အမည်", "Value": results_dict["၄. ဖခင်အမည် (Father's Name)"]},
        {"Field": "မှတ်ပုံတင်အမှတ်", "Value": results_dict["၅. မှတ်ပုံတင်အမှတ် (NRC No)"]},
        {"Field": "မွေးသကရာဇ်", "Value": results_dict["၆. မွေးသကရာဇ် (Date of Birth)"]},
        {"Field": "လူမျိုး/ဘာသာ", "Value": results_dict["၇. လူမျိုး/ဘာသာ (Nationality/Religion)"]},
        {"Field": "အရပ်", "Value": results_dict["၈. အရပ် (Height)"]},
        {"Field": "ထင်ရှားသည့်အမှတ်အသား", "Value": results_dict["၉. ထင်ရှားသည့်အမှတ်အသား (Identifying Mark)"]},
        {"Field": "ယုံကြည်မှု အမှတ်", "Value": results_dict["AI Extraction Confidence (0.0 - 1.0)"]},
    ])

    csv_buffer = BytesIO()
    # CRITICAL: Ensure UTF-8 encoding for Burmese characters in CSV
    df.to_csv(csv_buffer, index=False, encoding='utf-8')
    csv_content = csv_buffer.getvalue()

    # 4. Prepare DOC content (tab-separated for easy copy-paste)
    doc_content = "\n".join([f"{key}\t{value}" for key, value in results_dict.items()])

    return txt_content, csv_content, doc_content, results_dict


# --- 4. UI and Execution Flow ---

def process_image_and_display(original_image_pil, unique_key_suffix):
    """Performs AI extraction and displays results."""

    st.subheader("ပုံကို စစ်ဆေးနေပါသည်...")

    with st.spinner("AI အချက်အလက်များ ထုတ်ယူခြင်း (မြန်မာဘာသာ သီးသန့်)..."):
        time.sleep(0.5) # Slight delay for better UX
        raw_extracted_data = run_structured_extraction(original_image_pil)

        if raw_extracted_data is None:
             # Error handled within run_structured_extraction
             return

        # Prepare data for display/download
        txt_file, csv_file, doc_file, extracted_data = create_downloadable_files(raw_extracted_data)

    st.success(f"ထုတ်ယူမှု ပြီးစီးပါပြီ! ယုံကြည်မှု အမှတ်: **{extracted_data['AI Extraction Confidence (0.0 - 1.0)']}**")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("တင်ထားသော ပုံ")
        st.image(original_image_pil, use_column_width=True)

    with col2:
        st.header("ထုတ်ယူရရှိသော အချက်အလက်များ")

        # --- Results Form ---
        form_key = f"results_form_{unique_key_suffix}"
        with st.form(form_key):
            # Using extracted_data dictionary directly for easier maintenance
            st.text_input("အမှတ် (NRC Serial)", value=extracted_data["၁. အမှတ် (NRC Serial)"])
            st.text_input("ရက်စွဲ (Issue Date)", value=extracted_data["၂. ရက်စွဲ (Issue Date)"])
            st.text_input("အမည် (Name)", value=extracted_data["၃. အမည် (Name)"])
            st.text_input("ဖခင်အမည် (Father's Name)", value=extracted_data["၄. ဖခင်အမည် (Father's Name)"])
            st.text_input("မှတ်ပုံတင်အမှတ် (NRC No)", value=extracted_data["၅. မှတ်ပုံတင်အမှတ် (NRC No)"])
            st.text_input("မွေးသကရာဇ် (Date of Birth)", value=extracted_data["၆. မွေးသကရာဇ် (Date of Birth)"])
            st.text_input("လူမျိုး/ဘာသာ (Nationality/Religion)", value=extracted_data["၇. လူမျိုး/ဘာသာ (Nationality/Religion)"])
            st.text_input("အရပ် (Height)", value=extracted_data["၈. အရပ် (Height)"])
            st.text_input("ထင်ရှားသည့်အမှတ်အသား (Identifying Mark)", value=extracted_data["၉. ထင်ရှားသည့်အမှတ်အသား (Identifying Mark)"])
            st.text_input("ယုံကြည်မှု အမှတ် (Confidence)", value=extracted_data["AI Extraction Confidence (0.0 - 1.0)"])
            st.form_submit_button("အတည်ပြုမည်")


        st.subheader("အချက်အလက်များကို ဒေါင်းလုဒ်လုပ်ရန်")

        # --- Download Buttons ---
        st.download_button(
            label="⬇️ CSV ဖြင့် ဒေါင်းလုဒ်လုပ်ရန်",
            data=csv_file,
            file_name=f"nrc_myanmar_data_{unique_key_suffix}.csv",
            mime="text/csv",
            key=f"download_csv_{unique_key_suffix}"
        )
        st.download_button(
            label="⬇️ Plain Text ဖြင့် ဒေါင်းလုဒ်လုပ်ရန်",
            data=txt_file.encode('utf-8'), # Encode explicitly for text file
            file_name=f"nrc_myanmar_data_{unique_key_suffix}.txt",
            mime="text/plain",
            key=f"download_txt_{unique_key_suffix}"
        )
        st.download_button(
            label="⬇️ Word (.doc) ဖြင့် ဒေါင်းလုဒ်လုပ်ရန်",
            data=doc_file.encode('utf-8'), # Encode explicitly for doc file
            file_name=f"nrc_myanmar_data_{unique_key_suffix}.doc",
            mime="application/msword",
            key=f"download_doc_{unique_key_suffix}"
        )

# --- Main App Body ---

st.title("🪪 မြန်မာ မှတ်ပုံတင် ထုတ်ယူခြင်း (AI OCR)")
st.caption("AI ကို အသုံးပြု၍ မြန်မာမှတ်ပုံတင်ကတ်မှ အချက်အလက်အားလုံးကို **မြန်မာဘာသာ (Burmese) သီးသန့်** ထုတ်ယူခြင်း။")

# Generate a unique key suffix for session management
current_time_suffix = str(int(time.time()))

# --- Tab Setup ---
tab1, tab2 = st.tabs(["📷 ပုံရိုက်ယူခြင်း", "⬆️ ပုံတင်ခြင်း"])

# --- Live Capture Tab ---
with tab1:
    st.header("ကတ်ကို တိုက်ရိုက် ရိုက်ယူခြင်း")
    st.write("သင်၏ကင်မရာကို အသုံးပြု၍ မှတ်ပုံတင်ကတ်၏ မျက်နှာစာကို ရှင်းလင်းစွာ ရိုက်ယူပါ။")
    captured_file = st.camera_input("မှတ်ပုံတင်ကတ်ကို ရှင်းလင်းစွာ ထားရှိပြီး 'Take Photo' ကို နှိပ်ပါ", key="camera_input")

    if captured_file is not None:
        image_pil = handle_file_to_pil(captured_file)

        if image_pil is not None:
            # Check if this state has already been processed to prevent re-running on every click
            if 'last_processed_camera' not in st.session_state or st.session_state.last_processed_camera != captured_file:
                 st.session_state.last_processed_camera = captured_file
                 process_image_and_display(
                    image_pil,
                    f"live_{current_time_suffix}"
                 )
        else:
            st.error("ရိုက်ယူထားသော ပုံကို ဖတ်ရန် မအောင်မြင်ပါ။ ကျေးဇူးပြု၍ ကင်မရာ ရိုက်ယူမှု အောင်မြင်ကြောင်း သေချာပါစေ။")

# --- Upload File Tab ---
with tab2:
    st.header("ပုံဖိုင် တင်သွင်းခြင်း")
    st.write("မှတ်ပုံတင်ကတ်၏ မျက်နှာစာကို ရှင်းလင်းစွာ ရိုက်ထားသော ဓာတ်ပုံ သို့မဟုတ် စကင်ဖိုင်ကို တင်ပါ။")
    uploaded_file = st.file_uploader("မှတ်ပုံတင် ပုံတင်ရန်", type=['jpg', 'png', 'jpeg'], key="file_uploader")

    if uploaded_file is not None:
        image_pil = handle_file_to_pil(uploaded_file)

        if image_pil is not None:
            # Check if this state has already been processed
            if 'last_processed_upload' not in st.session_state or st.session_state.last_processed_upload != uploaded_file.name:
                st.session_state.last_processed_upload = uploaded_file.name
                process_image_and_display(
                    image_pil,
                    f"upload_{current_time_suffix}"
                )
        else:
            st.error("တင်ထားသော ပုံကို ဖတ်ရန် မအောင်မြင်ပါ။ ကျေးဇူးပြု၍ မှန်ကန်သော ပုံဖိုင် ဖြစ်ကြောင်း သေချာပါစေ။")