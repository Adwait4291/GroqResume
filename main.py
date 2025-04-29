# main.py 

import streamlit as st
from groq import Groq
import PyPDF2
import io
import os
import time
import logging
import json
import re
from dotenv import load_dotenv
from typing import Optional, Dict, List, Tuple, Any

# --- Configuration & Initialization ---

load_dotenv()

log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    force=True
)
logger = logging.getLogger(__name__)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama3-70b-8192")

MIN_JD_LENGTH = 100
MIN_RESUME_LENGTH = 150

# --- Groq Client Initialization ---

@st.cache_resource
def initialize_groq_client():
    """Initialize and return Groq client, handle missing API key."""
    if not GROQ_API_KEY:
        logger.error("GROQ_API_KEY environment variable not set.")
        return None
    try:
        logger.info("Initializing Groq client.")
        client = Groq(api_key=GROQ_API_KEY)
        return client
    except Exception as e:
        logger.exception("Failed to initialize Groq client.")
        return None

# --- Helper Functions ---

def extract_text_from_pdf(pdf_file_obj: io.BytesIO) -> Optional[str]:
    """Extract text from an uploaded PDF file object (BytesIO)."""
    # (Function remains the same as previous version)
    extracted_text = ""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file_obj)
        num_pages = len(pdf_reader.pages)
        logger.info(f"Reading PDF with {num_pages} pages.")
        for i, page in enumerate(pdf_reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    extracted_text += page_text + "\n"
                else:
                     logger.warning(f"No text extracted from page {i+1}.")
            except Exception as page_e:
                 logger.warning(f"Could not extract text from page {i+1}: {page_e}")
        if not extracted_text:
             logger.warning("No text extracted from any page of the PDF.")
             st.warning("Could not extract any text from the PDF. It might be image-based or corrupted.")
             return None
        logger.info(f"Successfully extracted text from PDF (approx {len(extracted_text)} chars).")
        cleaned_text = re.sub(r'\n\s*\n', '\n\n', extracted_text).strip()
        return cleaned_text
    except PyPDF2.errors.PdfReadError as e:
        st.error(f"Error reading PDF: Invalid or potentially encrypted PDF file. ({e})")
        logger.error(f"PyPDF2 PdfReadError: {e}", exc_info=True)
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during PDF text extraction: {e}")
        logger.exception("Unexpected error extracting PDF text.")
        return None


def analyze_resume_groq(_client: Groq, resume_text: str, job_description: str) -> Optional[Dict[str, Any]]:
    """Analyze resume against job description using Groq API, expecting JSON output."""
    # (Function remains the same as previous version)
    if not _client:
        st.error("Groq client not initialized.")
        return None

    if len(resume_text) < MIN_RESUME_LENGTH or len(job_description) < MIN_JD_LENGTH:
         st.warning("Resume or Job Description text is too short for meaningful analysis.")
         logger.warning("Analysis skipped due to short input text.")
         return None

    # Prompt requesting JSON with specific keys
    prompt = f"""
    Analyze the following resume against the provided job description.
    Provide a detailed, critical, and constructive analysis.

    **Resume Text:**
    ```text
    {resume_text}
    ```

    **Job Description:**
    ```text
    {job_description}
    ```

    **Instructions:**
    Respond ONLY with a valid JSON object. Do not include any text before or after the JSON object.
    The JSON object must contain the following keys:
    - "match_score": An integer score from 0 to 100 representing the overall alignment, considering skills, experience, keywords, and qualifications. Be realistic.
    - "score_rationale": A brief string explaining the main reasons for the given match_score.
    - "key_qualifications_match": A string summarizing how well the resume meets the *most critical* qualifications mentioned in the JD (use bullet points within the string, e.g., using markdown like '* Requirement: Matched/Partially Matched/Missing - Justification').
    - "missing_skills_requirements": A list of strings detailing important skills, tools, technologies, certifications, or specific experiences mentioned in the JD but *clearly missing* or insufficiently detailed in the resume. Be specific.
    - "strengths": A list of strings highlighting the resume's *most relevant* strengths for *this specific* job description (e.g., specific achievements, unique skill combinations, strong experience alignment).
    - "areas_for_improvement": A list of strings suggesting specific, actionable areas where the resume could be improved to better match *this* JD (focus on content, clarity, impact).
    - "suggested_resume_improvements": A list of strings providing concrete, actionable suggestions for *specific* changes or additions to the resume text. Examples: "Quantify achievement X by adding metrics like Y%", "Add keyword Z from the JD to the summary/skills", "Elaborate on project A experience focusing on B technology".
    - "keyword_analysis": An object containing ONLY one list: "missing_jd_keywords" (important keywords from JD not found in resume). Keep the list concise (max 5-7 keywords).

    Ensure all list values are strings. Ensure the entire output is a single, valid JSON object.
    Example keyword_analysis: {{ "missing_jd_keywords": ["Data Visualization", "Agile Methodology", "Cloud Platform X"] }}
    """

    try:
        start_time = time.time()
        logger.info(f"Sending request to Groq API with model {GROQ_MODEL}...")
        response = _client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert ATS (Applicant Tracking System) and human recruiter resume analyzer. You provide critical, actionable feedback. Respond ONLY with the requested JSON object."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_tokens=4096,
        )
        end_time = time.time()
        api_duration = end_time - start_time
        logger.info(f"Groq API response received in {api_duration:.2f} seconds.")

        response_content = response.choices[0].message.content
        logger.debug(f"Raw Groq response content start:\n{response_content[:500]}...")

        # Robust JSON Parsing
        try:
            json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
            if json_match:
                json_string = json_match.group(0)
                analysis_json = json.loads(json_string)
                logger.info("Successfully parsed JSON response from Groq.")
            else:
                st.error("Could not find a valid JSON object in the AI response.")
                logger.error(f"No JSON object found in Groq response. Raw response was:\n{response_content}")
                return None

            # Schema Validation
            required_keys = ["match_score", "score_rationale", "key_qualifications_match",
                             "missing_skills_requirements", "strengths", "areas_for_improvement",
                             "suggested_resume_improvements", "keyword_analysis"]
            missing_keys = [key for key in required_keys if key not in analysis_json]
            if "keyword_analysis" in analysis_json and "missing_jd_keywords" not in analysis_json.get("keyword_analysis", {}): # Safer check
                 missing_keys.append("keyword_analysis.missing_jd_keywords")
                 if isinstance(analysis_json.get("keyword_analysis"), dict):
                     analysis_json["keyword_analysis"]["missing_jd_keywords"] = []

            if missing_keys:
                logger.warning(f"Groq response JSON missing expected keys: {missing_keys}. Response: {analysis_json}")
                st.warning(f"Analysis response might be incomplete. Missing fields: {', '.join(missing_keys)}")
                for key in missing_keys:
                    if key == "keyword_analysis.missing_jd_keywords":
                        if "keyword_analysis" not in analysis_json: analysis_json["keyword_analysis"] = {}
                        if "missing_jd_keywords" not in analysis_json["keyword_analysis"]: analysis_json["keyword_analysis"]["missing_jd_keywords"] = []
                        continue
                    if key == "keyword_analysis": analysis_json[key] = {"missing_jd_keywords": []}
                    elif key in ["missing_skills_requirements", "strengths", "areas_for_improvement", "suggested_resume_improvements"]: analysis_json[key] = []
                    else: analysis_json[key] = "N/A"

            if not isinstance(analysis_json.get("match_score"), int):
                logger.warning("Match score is not an integer. Setting to N/A.")
                analysis_json["match_score"] = "N/A"

            return analysis_json

        except json.JSONDecodeError as json_e:
            st.error(f"Failed to parse the analysis response from the AI. Please check the format or try again. Error: {json_e}")
            logger.error(f"Failed to decode Groq response as JSON. Raw response was:\n{response_content}", exc_info=True)
            return None
        except Exception as parse_e:
            st.error(f"An error occurred while processing the AI response: {parse_e}")
            logger.error(f"Error processing AI response: {parse_e}. Raw response was:\n{response_content}", exc_info=True)
            return None

    except Exception as e:
        st.error(f"An error occurred during analysis with the Groq API: {e}")
        logger.exception("Error during Groq API call.")
        return None


# --- Streamlit UI ---

st.set_page_config(
    page_title="Resume Analyzer Pro",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Main Application Logic ---
def main():
    st.title("✨ Resume Analyzer Pro")
    st.subheader("Get AI-powered feedback to tailor your resume effectively")
    st.markdown("---") # Use markdown for horizontal rule

    client = initialize_groq_client()
    if not client:
        st.error("Groq client could not be initialized. Please ensure the GROQ_API_KEY is set correctly in your environment variables (.env file).")
        st.stop()

    # Session State Initialization
    if "resume_text" not in st.session_state: st.session_state.resume_text = None
    if "job_description" not in st.session_state: st.session_state.job_description = ""
    if "analysis_result" not in st.session_state: st.session_state.analysis_result = None
    if "analysis_requested" not in st.session_state: st.session_state.analysis_requested = False
    if "last_upload_name" not in st.session_state: st.session_state.last_upload_name = None

    # --- Inputs Area ---
    # Using st.form to ensure inputs are submitted together with the button
    # This prevents accidental reruns just from editing text areas
    with st.form(key="input_form"):
        col1, col2 = st.columns([0.6, 0.4])

        with col1:
            st.markdown("#### 1. Upload Your Resume")
            uploaded_file = st.file_uploader(
                "Upload Resume (PDF only)",
                type=['pdf'],
                key="resume_upload_widget", # Give it a specific key inside the form
                help="Ensure your PDF contains selectable text."
            )
            # Display extracted text status outside the form if available
            if st.session_state.resume_text:
                 with st.expander("View Extracted Resume Text"):
                      st.text_area("Extracted Text", st.session_state.resume_text, height=150, disabled=True, key="resume_extracted_text_display_form")

        with col2:
            st.markdown("#### 2. Paste Job Description")
            job_description_input = st.text_area( # Use a different variable name inside the form
                "Paste the full job description:",
                value=st.session_state.job_description, # Initialize with session state
                height=300,
                key="jd_paste_widget", # Give it a specific key inside the form
                placeholder="Include all requirements..."
            )

        st.markdown("---")
        st.markdown("#### 3. Run Analysis")
        # The submit button for the form IS the analysis trigger
        submit_button = st.form_submit_button(
            label="🚀 Analyze Now",
            type="primary",
            use_container_width=True
            # Disabled state handled implicitly by checking inputs after submission
        )

    # --- Process Inputs and Trigger Analysis ONLY on Form Submission ---
    if submit_button:
        # --- Handle File Upload within Form Submission ---
        if uploaded_file is not None:
            # Check if it's a new file or the same one re-submitted
            if uploaded_file.name != st.session_state.last_upload_name or st.session_state.resume_text is None:
                 st.session_state.analysis_result = None # Clear old results
                 st.session_state.analysis_requested = False
                 st.session_state.last_upload_name = uploaded_file.name
                 with st.spinner("Extracting text from PDF..."):
                      bytes_data = uploaded_file.getvalue()
                      st.session_state.resume_text = extract_text_from_pdf(io.BytesIO(bytes_data))
                      if st.session_state.resume_text: st.success("Resume text extracted.")
                      else: st.error("Failed to extract text from new PDF.")
            # If same file re-submitted, resume_text should still be in session state
        elif not st.session_state.resume_text: # No file uploaded and no text in session state
             st.error("Please upload a resume PDF.")


        # Update job description in session state from the form input
        st.session_state.job_description = job_description_input

        # --- Validation and Analysis Trigger ---
        st.session_state.analysis_requested = True # Mark that analysis was attempted
        st.session_state.analysis_result = None # Clear previous results

        # Validate inputs *after* form submission
        valid_jd = len(st.session_state.job_description) >= MIN_JD_LENGTH
        valid_resume = st.session_state.resume_text is not None and len(st.session_state.resume_text) >= MIN_RESUME_LENGTH

        if not valid_resume:
             st.error(f"Resume text missing or too short (needs > {MIN_RESUME_LENGTH} chars). Cannot analyze.")
        if not valid_jd:
             st.error(f"Job description is too short (needs > {MIN_JD_LENGTH} chars). Cannot analyze.")

        if valid_resume and valid_jd:
            with st.spinner("🤖 Performing AI analysis via Groq... Please wait."):
                analysis_start_time = time.time()
                st.session_state.analysis_result = analyze_resume_groq(
                    client,
                    st.session_state.resume_text,
                    st.session_state.job_description
                )
                analysis_end_time = time.time()
                logger.info(f"Total analysis process time (including API call): {analysis_end_time - analysis_start_time:.2f} seconds")
        else:
             st.session_state.analysis_requested = False # Reset if validation failed


    # --- Display Analysis Results ---
    # Display results if analysis was successful (result is not None)
    if st.session_state.analysis_result:
        results = st.session_state.analysis_result
        st.header("📊 Analysis Results")

        # Score & Rationale using st.metric
        score = results.get("match_score")
        rationale = results.get("score_rationale", "N/A")
        score_label = f"{score}/100" if isinstance(score, int) else str(score)
        st.metric(label="Overall Match Score", value=score_label)
        if rationale != "N/A": st.caption(f"**Rationale:** {rationale}")
        if isinstance(score, int): st.progress(score / 100)
        st.markdown("---")

        # Detailed Sections
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            with st.container(border=True):
                st.subheader("🔑 Key Qualifications Match")
                st.markdown(results.get("key_qualifications_match", "_Not provided_"))
            with st.container(border=True):
                st.subheader("✅ Strengths")
                strengths = results.get("strengths", [])
                if strengths:
                    for item in strengths: st.markdown(f"- {item}")
                else: st.info("_No specific strengths highlighted._")
            with st.container(border=True):
                st.subheader("🔑 Keyword Analysis")
                kw_analysis = results.get("keyword_analysis", {})
                missing_kw = kw_analysis.get("missing_jd_keywords", [])
                st.markdown("**Keywords from JD Missing in Resume:**")
                if missing_kw: st.info(f"{', '.join(missing_kw)}")
                else: st.success("_No critical missing keywords identified._")

        with res_col2:
             with st.container(border=True):
                 st.subheader("❌ Missing Skills/Requirements")
                 missing = results.get("missing_skills_requirements", [])
                 if missing:
                      for item in missing: st.markdown(f"- {item}")
                 else: st.success("_No critical missing skills or requirements identified._")
             with st.container(border=True):
                st.subheader("📉 Areas for Improvement")
                areas = results.get("areas_for_improvement", [])
                if areas:
                    for item in areas: st.markdown(f"- {item}")
                else: st.info("_No specific areas for improvement highlighted._")

        # Suggested Improvements (Full Width)
        with st.container(border=True):
             st.subheader("💡 Suggested Resume Improvements")
             suggestions = results.get("suggested_resume_improvements", [])
             if suggestions:
                  for item in suggestions: st.markdown(f"- {item}")
             else: st.info("_No specific suggestions provided._")

        # --- REMOVED Download Button ---
        st.markdown("---")
        # The st.download_button(...) call has been deleted.


    # Show message if button was clicked but analysis failed
    elif st.session_state.analysis_requested:
         # Error messages should be displayed during the analysis trigger phase
         # This is a fallback message
         if not st.session_state.analysis_result: # Check if result is actually None after request
              st.error("Analysis could not be completed. Please review inputs or error messages above.")


    # --- Tips Expander ---
    st.markdown("---") # Separator before tips
    with st.expander("💡 Tips and Information"):
        st.markdown("""
        * **Accuracy:** The AI analysis provides suggestions based on patterns and the text provided. Critically evaluate the feedback before applying it.
        * **Formatting:** Ensure your uploaded PDF has selectable text for best results. Image-based PDFs cannot be read accurately.
        * **Context:** The quality of the analysis heavily depends on the detail and clarity of both your resume and the job description.
        * **Keywords:** Pay attention to the keyword analysis. ATS systems often rely heavily on keyword matching.
        * **Tailoring:** Remember to tailor your resume for *each specific job application* for maximum impact.
        """)

if __name__ == "__main__":
    main()