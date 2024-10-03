from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from flask_session import Session
import os
import model  # Importing functions from your model.py
from flask_cors import CORS

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Enable CORS
CORS(app)

# Configure session to use filesystem (instead of Redis)
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = 'flask_session_files'
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True

# Ensure the directory for session files exists
os.makedirs(app.config['SESSION_FILE_DIR'], exist_ok=True)

Session(app)  # Initialize the session for Flask app


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        resume_text = ""
        job_description_text = ""

        # Check if the 'resume_file' part is present in the request files
        if 'resume_file' in request.files and request.files['resume_file'].filename != '':
            file = request.files['resume_file']
            if file.filename.endswith('.pdf'):
                resume_text = model.pdf_extractor(file)
            elif file.filename.endswith('.docx'):
                resume_text = model.docx_extractor(file)
            session['resume_text'] = resume_text

        # Similarly, check for the job description file
        if 'job_description_file' in request.files and request.files['job_description_file'].filename != '':
            file = request.files['job_description_file']
            if file.filename.endswith('.pdf'):
                job_description_text = model.pdf_extractor(file)
            elif file.filename.endswith('.docx'):
                job_description_text = model.docx_extractor(file)
            session['job_description_text'] = job_description_text

        # Handle text input as well
        if 'resume_text' in request.form and request.form['resume_text'].strip():
            session['resume_text'] = request.form['resume_text']
        if 'job_description_text' in request.form and request.form['job_description_text'].strip():
            session['job_description_text'] = request.form['job_description_text']

        operation = request.form.get('operation')

        # Processing based on the operation
        if operation == 'suggest_improvements':
            # Get suggestions and analysis results
            improvement_suggestions = model.suggest_improvements(
                session.get('resume_text', ''), session.get('job_description_text', ''))
            analysis_results = model.analyze_resume(
                session.get('resume_text', ''), session.get('job_description_text', ''))

            # Return only the necessary data for "Suggest Improvements"
            results = {
                "match_percentage": analysis_results.get('match_percentage', 0),
                "improvements": improvement_suggestions,
                "common_keywords": analysis_results.get('common_keywords', {})
            }


        elif operation == 'google_gemini':
            results = model.generate_response_from_gemini(
                session.get('resume_text', ''), session.get('job_description_text', ''))
            print("Google Gemini Results: ", results)  # Debug print statement

            
        elif operation == 'combine_results':
            results = model.combined_results(
                session.get('resume_text', ''), session.get('job_description_text', ''))
        
        elif operation == 'check_grammar':
            highlighted_text, error_details, readability_scores = model.identify_and_highlight_errors(
                session.get('resume_text', ''))
            
            results = {
                "highlighted_text": highlighted_text,
                "error_details": error_details,
                "readability_scores": readability_scores
            }

        else:
            results = {}

        return jsonify(results)

    return render_template('index.html')


@app.route('/reset', methods=['GET'])
def reset():
    session.clear()
    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True)
