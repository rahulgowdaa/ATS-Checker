
# ATS Checker

## Overview
ATS Checker is a web-based application that helps users compare their resumes with job descriptions. The tool analyzes both documents and provides improvement suggestions to enhance the chances of a resume passing through Applicant Tracking Systems (ATS). Additionally, the application utilizes Google Gemini AI to analyze the match between a resume and job description, and provides feedback on missing skills and areas of improvement.

## Features
- **Suggest Improvements**: Compare your resume with the job description and receive suggestions for improvement.
- **Google Gemini AI Integration**: Get in-depth analysis of your resume and job description, along with missing keywords and candidate summaries.

## Project Structure

```bash
ATS-Checker/
├── app.py                  # Main Flask application
├── config.py               # Configuration file for app settings
├── model.py                # Contains core logic for resume analysis and Google Gemini API integration
├── requirements.txt        # List of dependencies and libraries required
├── README.md               # Documentation (this file)
├── .gitignore              # Files and directories to be ignored by Git
├── flask_session_files/     # Session data files (ignored by git)
├── Templates/              # Contains HTML files for rendering views
│   └── index.html          # Main HTML page for the web application
├── static/                 # Contains static files like JS, CSS, and images
│   ├── script.js           # JavaScript for client-side logic
│   ├── styles.css          # Custom styles for the web application
├── venv/                   # Virtual environment for the project (ignored by git)
└── key2.json               # (Example) Credentials for Google Gemini API
```
## Prerequisites
Before you begin, ensure you have the following installed:
- Python 3.10 or higher
- Flask
- Git
- Virtualenv (optional)

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/rahulgowdaa/ATS-Checker.git
   cd ATS-Checker
   ```

2. **Set up a virtual environment (optional but recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3. **Install the required dependencies:**
    ```bash 
    pip install -r requirements.txt
    ```
4. **Set up your environment variables (Optional):** 
You may want to configure environment variables for sensitive data like API keys (e.g., for Google Gemini).

5. **Run the application:**
    ```bash
    python app.py
    ```
6. **Open your web browser and navigate to:**
    ```bash
    http://127.0.0.1:5000/
    ```

## Usage
1. **Upload a Resume:** Upload a resume in PDF or DOCX format, or paste the resume text into the provided input box.
2. **Upload Job Description:** Upload a job description in PDF or DOCX format, or paste the job description text.
3. **Choose an Operation:** Select one of the following options:
- Suggest Improvements: Receive improvement suggestions.
- Google Gemini AI: Get detailed feedback on your resume using Google Gemini.
4. **Submit:** Click "Submit" to receive the results.
5. **Results:** View the suggestions, missing keywords, and candidate summary.


## API Integration
- **Google Gemini AI:** The application integrates with Google Gemini AI to analyze resumes and job descriptions.

## Contributing
1. **Fork the Repository:** Click on the "Fork" button on the top-right corner of the repo page.
2. **Clone your Fork:** Clone your forked repository to your local machine.
    ```bash 
    git clone https://github.com/your-username/ATS-Checker.git
    ```
3. **Create a Branch:** Create a new branch for your feature or fix.
    ```bash
    git checkout -b feature/new-feature
    ```
4. **Commit Your Changes:** Make changes and commit them to your branch. 
    ```bash
    git commit -m "Add new feature -- feature name"
    ```
5. **Push to GitHub:** Push your branch to GitHub.
    ```bash
    git push origin feature/new-feature
    ```
6. **Submit a Pull Request:** Go to your fork on GitHub and create a pull request.

## Acknowledgements

- Google Gemini API for AI-based analysis.
- LanguageTool for grammar and spelling checks.
- Flask for providing the web framework.

## Contact
For any questions or issues, feel free to reach out:
- Rahul Gowda Aswatha: rahulgowda20799@gmail.com