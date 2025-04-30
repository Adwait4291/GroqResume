# GroqResume

APP LINK : https://groqresume-wsow7fvgc6vkelsr4anuqk.streamlit.app/

# GroqResume: AI-Powered Resume Analyzer Pro ✨

## Overview

GroqResume is an interactive web application built with Streamlit that leverages the power of Large Language Models (LLMs) via the ultra-fast Groq LPU™ Inference Engine. It allows users to upload their resume (in PDF format) and paste a job description to receive instant, detailed feedback on how well the resume aligns with the role. The analysis includes a match score, strengths, weaknesses, keyword analysis, and actionable suggestions for improvement, helping users tailor their resumes effectively for specific job applications.

## Features

* **PDF Resume Upload:** Accepts resume uploads in PDF format.
* **Text Extraction:** Extracts text content from uploaded PDFs using PyPDF2.
* **Job Description Input:** Provides a text area for users to paste the target job description.
* **AI-Powered Analysis:** Utilizes a powerful LLM (Llama 3 via Groq by default) to perform an in-depth comparison between the resume and job description.
* **Structured Feedback:** Delivers analysis in a structured format, including:
    * Overall Match Score (0-100)
    * Rationale for the score
    * Analysis of Key Qualifications Match
    * Identified Strengths relevant to the job
    * Missing Skills & Requirements from the JD
    * General Areas for Resume Improvement
    * Specific, Actionable Suggestions for Resume Edits
    * Keyword Analysis (important JD keywords missing in the resume)
* **Fast Inference:** Leverages the Groq API for remarkably fast analysis results. (Based on Groq's known capabilities)
* **Interactive Web UI:** Built with Streamlit for an easy-to-use and responsive user interface.

## Technology Stack

* **Language:** Python 3.x
* **Web Framework:** Streamlit - For building the interactive user interface. (Streamlit allows turning data scripts into shareable web apps) (Search result [2.1], [2.3])
* **AI Model:** Llama 3 (specifically `llama3-70b-8192` by default via Groq API) - A large language model developed by Meta. (Search result [3.1])
* **AI Inference:** Groq LPU™ Inference Engine - Provides high-speed LLM inference through its custom hardware (Language Processing Units). (Search result [3.1], [3.3])
* **PDF Processing:** PyPDF2 - A Python library for extracting text content from PDF files.
* **API Client:** `groq` Python library - For interacting with the GroqCloud API.
* **Configuration:** `python-dotenv` - For managing environment variables (like API keys) during local development.
* **Standard Libraries:** `json`, `re`, `os`, `logging`, `io`, `time`, `typing`.

## AI/LLM Integration & Techniques

This project utilizes several AI and LLM techniques:

1.  **Prompt Engineering:** A detailed system prompt guides the Llama 3 model to act as an expert ATS/recruiter. It specifies the desired analysis points and crucially instructs the model to return its findings *only* in a structured JSON format. This is key to reliably extracting and displaying the analysis results in the UI.
2.  **Structured Output Generation:** The core analysis relies on the LLM's ability to generate a JSON object adhering to a predefined schema (match score, strengths, missing skills, etc.). Robust parsing logic (using regex and `json.loads`) is implemented in Python to handle the LLM's response.
3.  **Zero-Shot Analysis:** The application performs resume analysis in a zero-shot manner. The LLM understands and executes the task based on the instructions in the prompt and the provided resume/JD text, without needing specific fine-tuning on resume data beforehand.
4.  **High-Speed Inference:** By using the Groq API, the application benefits from significantly reduced latency for the LLM response compared to traditional GPU-based inference, leading to a better user experience.

## Complexities & Challenges

* **PDF Text Extraction Variability:** Extracting text accurately from diverse PDF layouts can be challenging. PyPDF2 may struggle with complex formatting, multi-column layouts, tables, images-as-text (scanned PDFs), or password-protected/corrupted files. The application relies on the PDF containing selectable text. (Common issues include formatting loss, incorrect character encoding, inability to read scanned images) (Search result [4.1], [4.2], [4.3], [4.4]). Basic error handling is included.
* **Ensuring Valid JSON from LLM:** While the prompt strongly requests JSON, LLMs can sometimes fail to adhere perfectly, occasionally adding introductory text or deviating from the requested schema. The code uses regular expressions (`re`) to isolate the JSON block within the response and includes validation checks for required keys to handle potential inconsistencies.
* **Prompt Robustness:** Crafting a prompt that consistently yields high-quality, relevant, and correctly formatted analysis across different resumes and job descriptions requires careful design and iteration.
* **API Key Security:** Managing the `GROQ_API_KEY` securely is crucial. The use of `.env` files for local development and Streamlit Cloud's secrets management for deployment is recommended. Ensure `.env` and `.streamlit/secrets.toml` are in `.gitignore`.
* **Input Quality Dependency:** The quality and usefulness of the AI analysis heavily depend on the clarity, detail, and content quality of both the input resume and the job description.

## Setup and Local Usage

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd GroqResume-1
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
   
4.  **Set up API Key:**
    * Obtain an API key from [GroqCloud](https://console.groq.com/keys).
    * Create a file named `.env` in the project root directory (`GroqResume-1`).
    * Add your API key to the `.env` file:
        ```env
        GROQ_API_KEY="your_actual_groq_api_key"
        # Optional: Specify a different model if needed
        # GROQ_MODEL="llama3-8b-8192"
        # Optional: Set log level (e.g., DEBUG, INFO, WARNING)
        # LOG_LEVEL="DEBUG"
        ```
    * **Important:** Add `.env` to your `.gitignore` file to prevent committing your key.
5.  **Run the Streamlit app:**
    ```bash
    streamlit run main.py
    ```
   
6.  Open your browser to the local URL provided by Streamlit (usually `http://localhost:8501`).

## Deployment

This application is designed to be easily deployable on **Streamlit Community Cloud**:

1.  Ensure your code is pushed to a GitHub repository (including `requirements.txt` but *excluding* `.env`).
2.  Create a `.streamlit/secrets.toml` file locally (and add it to `.gitignore`) with your Groq key:
    ```toml
    # .streamlit/secrets.toml
    GROQ_API_KEY = "your_actual_groq_api_key"
    ```
3.  Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) using your GitHub account.
4.  Click "New app", select your repository, branch, and set `main.py` as the main file path.
5.  In the "Advanced settings", paste the contents of your local `secrets.toml` file into the "Secrets" text box.
6.  Click "Deploy!".

(Refer to the [Streamlit Deployment Docs](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app) for more details).

## Future Improvements (Optional)

* Add support for other file formats (e.g., .docx).
* Implement OCR for image-based PDFs.
* Allow users to select different LLM models available via Groq.
* Provide options to save/export the analysis results.
* Add user accounts or session persistence for history.
* Refine prompt engineering further based on user feedback.
