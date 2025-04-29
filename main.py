import streamlit as st
from groq import Groq
import PyPDF2
import io
import os
import time
import logging
import json               # Import json for parsing
import tempfile           # Import tempfile for safer temporary handling (if needed)
from dotenv import load_dotenv
from typing import Optional, Dict, List, Tuple

# --- Configuration & Initialization ---

# Load environment variables (.env file is optional)
load_dotenv()

# Logging Configuration
log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Get Groq API Key and Model from environment variables
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama3-8b-8192") # Default to llama3 if not set

# --- Groq Client Initialization ---

def initialize_groq_client():
    """Initialize and return Groq client, handle missing API key."""
    if not GROQ_API_KEY:
        st.error("GROQ_API_KEY environment variable not set. Please configure it.")
        logger.error("GROQ_API_KEY environment variable not set.")
        return None
    try:
        return Groq(api_key=GROQ_API_KEY)
    except Exception as e:
        st.error(f"Failed to initialize Groq client: {e}")
        logger.exception("Failed to initialize Groq client.")
        return None

# --- Helper Functions ---

def extract_text_from_pdf(pdf_file_obj: io.BytesIO) -> Optional[str]:
    """Extract text from an uploaded PDF file object (BytesIO)."""
    try:
        # PyPDF2 works directly with file-like objects
        pdf_reader = PyPDF2.PdfReader(pdf_file_obj)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n" # Add newline between pages
        logger.info(f"Successfully extracted text from PDF (approx {len(text)} chars).")
        return text
    except PyPDF2.errors.PdfReadError as e:
        st.error(f"Error reading PDF: Invalid or corrupted PDF file. ({e})")
        logger.error(f"PyPDF2 PdfReadError: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during PDF text extraction: {e}")
        logger.exception("Unexpected error extracting PDF text.")
        return None

def analyze_resume_groq(client: Groq, resume_text: str, job_description: str) -> Optional[dict]:
    """Analyze resume against job description using Groq API, expecting JSON output."""
    if not client:
        st.error("Groq client not initialized.")
        return None

    prompt = f"""
    Analyze the following resume against the provided job description.
    Provide a detailed analysis covering the points below.

    **Resume Text:**
    ```
    {resume_text}
    ```

    **Job Description:**
    ```
    {job_description}
    ```

    **Instructions:**
    Respond ONLY with a valid JSON object containing the following keys:
    - "match_score": An integer score from 0 to 100 representing the overall match.
    - "key_qualifications_match": A string summarizing how well the resume matches key qualifications (bullet points or paragraph).
    - "missing_skills_requirements": A list of strings detailing important skills or requirements mentioned in the JD but missing or not evident in the resume.
    - "strengths": A list of strings highlighting the resume's strengths in relation to the job description.
    - "areas_for_improvement": A list of strings suggesting specific areas where the resume could be improved for this role.
    - "suggested_resume_improvements": A list of strings providing concrete suggestions for specific changes or additions to the resume text.

    Ensure the output is ONLY the JSON object and nothing else.
    Example JSON structure:
    {{
      "match_score": 75,
      "key_qualifications_match": "Good alignment with core requirements like X and Y. Experience in Z is relevant.",
      "missing_skills_requirements": ["Specific Tool A", "Certification B", "Experience with process C"],
      "strengths": ["Strong background in X", "Demonstrated ability in Y", "Quantifiable achievement in Z project"],
      "areas_for_improvement": ["Quantify achievements further", "Tailor summary to the role", "Add details on tool A experience if possible"],
      "suggested_resume_improvements": ["Rephrase bullet point X to include metrics", "Add 'Proficient in Specific Tool A' to skills section if applicable", "Include a brief mention of project Z outcome in the summary"]
    }}
    """

    try:
        start_time = time.time()
        logger.info(f"Sending request to Groq API with model {GROQ_MODEL}...")
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert resume analyzer and career coach. Respond ONLY with the requested JSON object."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5, # Lower temperature for more deterministic JSON output
            max_tokens=3000, # Adjust as needed
            # Consider adding response_format for explicit JSON mode if supported by model/API
            # response_format={"type": "json_object"} # Example if supported
        )
        end_time = time.time()
        logger.info(f"Groq API response received in {end_time - start_time:.2f} seconds.")

        response_content = response.choices[0].message.content
        logger.debug(f"Raw Groq response content:\n{response_content}")

        # Attempt to parse the JSON response
        try:
            analysis_json = json.loads(response_content)
            # Basic validation of expected keys
            required_keys = ["match_score", "key_qualifications_match", "missing_skills_requirements",
                             "strengths", "areas_for_improvement", "suggested_resume_improvements"]
            if not all(key in analysis_json for key in required_keys):
                logger.warning(f"Groq response parsed as JSON but missing expected keys. Response: {analysis_json}")
                st.warning("Analysis received, but some expected fields might be missing.")
                # Return partial data or None depending on desired strictness
            return analysis_json
        except json.JSONDecodeError as json_e:
            st.error(f"Failed to parse the analysis response from the AI as JSON. Please try again. Error: {json_e}")
            logger.error(f"Failed to decode Groq response as JSON. Raw response was:\n{response_content}", exc_info=True)
            return None

    except Exception as e:
        st.error(f"An error occurred during analysis with the Groq API: {e}")
        logger.exception("Error during Groq API call.")
        return None

# --- Streamlit UI ---

# Configure Streamlit page
st.set_page_config(
    page_title="Resume Analyzer (Groq)",
    page_icon="📝",
    layout="wide"
)

# Custom CSS (can be kept or modified)
st.markdown("""
    <style>
    .stApp {
        /* max-width: 1200px; */ /* Removed max-width for potentially wider content */
        margin: 0 auto;
        padding: 1rem;
    }
    .analysis-section {
        padding: 15px;
        border-radius: 8px;
        background-color: #f0f2f6;
        margin-bottom: 15px;
        border: 1px solid #dcdcdc;
    }
    .analysis-section h3 {
        margin-top: 0;
        color: #0066cc;
        border-bottom: 1px solid #ddd;
        padding-bottom: 5px;
    }
    .match-score {
        font-size: 2em;
        font-weight: bold;
        color: #0066cc;
        text-align: center;
    }
    .score-box {
         padding: 20px;
         border-radius: 10px;
         background-color: #e7f0fa;
         text-align: center;
         margin-bottom: 15px;
    }
    li { margin-bottom: 0.5em; } /* Add space between list items */
    </style>
    """, unsafe_allow_html=True)

def main():
    st.title("📝 Resume Analyzer powered by Groq")
    st.write("Upload your resume (PDF) and paste the job description to get an AI-powered analysis.")

    # Initialize Groq client
    client = initialize_groq_client()
    if not client:
        st.warning("Groq client could not be initialized. Please check API key configuration.")
        # Optionally stop execution if client is essential: st.stop()

    # --- User Information Inputs (Optional) ---
    # Consider if these are needed without database storage
    with st.expander("Optional: Enter Your Information"):
        name = st.text_input("Name")
        email = st.text_input("Email")
        # contact_no = st.text_input("Contact Number") # Removed for privacy if not stored
        # city = st.text_input("City") # Removed for privacy if not stored
        # linkedin_profile = st.text_input("LinkedIn Profile URL") # Removed for privacy if not stored
        preferred_job_role = st.text_input("Preferred Job Role (Optional)")
        preferred_job_location = st.text_input("Preferred Job Location (Optional)")

    # --- Resume and Job Description Inputs ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. Upload Resume")
        uploaded_file = st.file_uploader("Upload your resume (PDF only)", type=['pdf'], key="resume_upload")
        # Placeholder for extracted text display
        resume_text_display = st.empty()

    with col2:
        st.subheader("2. Paste Job Description")
        job_description = st.text_area(
            "Paste the job description here:",
            height=300,
            key="jd_paste",
            placeholder="Paste the complete job description..."
        )

    st.divider()

    # --- Analysis Execution and Display ---
    analysis_result = None # Variable to store analysis output

    if uploaded_file is not None and job_description.strip():
        # Extract resume text immediately after upload for display
        # Use BytesIO to pass the content directly to the extraction function
        bytes_data = uploaded_file.getvalue()
        resume_text = extract_text_from_pdf(io.BytesIO(bytes_data))

        if resume_text:
            with resume_text_display.container():
                 with st.expander("View Extracted Resume Text", expanded=False):
                      st.text_area("Extracted Text", resume_text, height=200, disabled=True, key="resume_extracted_text")
        else:
            # Error message handled within extract_text_from_pdf
            pass # Don't proceed if text extraction failed

        # --- Analysis Button ---
        st.subheader("3. Analyze")
        if st.button("Analyze Resume", type="primary", disabled=(not resume_text)):
            if not resume_text:
                 st.error("Cannot analyze: Failed to extract text from resume PDF.")
            elif client is None:
                 st.error("Cannot analyze: Groq client is not available.")
            else:
                 with st.spinner("🤖 Calling Groq API for analysis... This may take a moment."):
                      # --- Call Analysis Function ---
                      analysis_result = analyze_resume_groq(client, resume_text, job_description)

    # --- Display Analysis Results ---
    if analysis_result:
        logger.info("Displaying analysis results.")
        st.header("📊 Analysis Results")

        # Display Score prominently
        score = analysis_result.get("match_score", "N/A")
        st.markdown('<div class="score-box">Match Score<div class="match-score">{}</div></div>'.format(f"{score}/100" if isinstance(score, int) else score), unsafe_allow_html=True)

        # Use columns for better layout of details
        res_col1, res_col2 = st.columns(2)

        with res_col1:
            # Key Qualifications Match
            with st.container(border=True):
                 st.subheader("🔑 Key Qualifications Match")
                 st.markdown(analysis_result.get("key_qualifications_match", "_Not provided_"))

            # Strengths
            with st.container(border=True):
                st.subheader("💪 Strengths")
                strengths = analysis_result.get("strengths", [])
                if strengths:
                    for item in strengths:
                        st.markdown(f"- {item}")
                else:
                    st.write("_No specific strengths highlighted in analysis._")

        with res_col2:
             # Missing Skills/Requirements
             with st.container(border=True):
                 st.subheader("❓ Missing Skills/Requirements")
                 missing = analysis_result.get("missing_skills_requirements", [])
                 if missing:
                     for item in missing:
                         st.markdown(f"- {item}")
                 else:
                     st.write("_No specific missing skills identified or all requirements met._")

            # Areas for Improvement
             with st.container(border=True):
                st.subheader("📉 Areas for Improvement")
                areas = analysis_result.get("areas_for_improvement", [])
                if areas:
                    for item in areas:
                        st.markdown(f"- {item}")
                else:
                    st.write("_No specific areas for improvement highlighted._")


        # Suggested Improvements (Full Width Below Columns)
        with st.container(border=True):
            st.subheader("💡 Suggested Resume Improvements")
            suggestions = analysis_result.get("suggested_resume_improvements", [])
            if suggestions:
                for item in suggestions:
                    st.markdown(f"- {item}")
            else:
                st.write("_No specific suggestions provided._")


        # --- Download Button ---
        try:
            # Prepare analysis text for download (pretty print JSON)
            analysis_text_for_download = json.dumps(analysis_result, indent=2)
            st.download_button(
                label="Download Full Analysis (JSON)",
                data=analysis_text_for_download,
                file_name="resume_analysis.json",
                mime="application/json"
            )
        except Exception as e:
             logger.error(f"Failed to prepare analysis for download: {e}")
             st.warning("Could not prepare analysis for download.")

    # --- Tips Expander (Keep as before) ---
    with st.expander("💡 Tips for Better Results"):
        st.markdown("""
        - Ensure your uploaded PDF contains selectable text (not just an image).
        - Paste the full, relevant job description.
        - Review the AI's suggestions and apply them critically to your resume.
        - Tailor your resume for *each specific job application*.
        """)

if __name__ == "__main__":
    main()