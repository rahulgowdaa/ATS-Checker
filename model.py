# model.py
from nltk.corpus import wordnet as wn
import spacy
from spacy.matcher import PhraseMatcher, Matcher
from collections import Counter, defaultdict
import re
import google.generativeai as genai
import fitz  # PyMuPDF
import docx
from io import BytesIO
import json
import textstat
import logging

import language_tool_python
from markupsafe import Markup

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

genai.configure(api_key="AIzaSyDNMw9q5Cndtg3WXDUc6JqnKkQoyTT4q3c")


# Load the spaCy model
nlp = spacy.load("en_core_web_md")


def pdf_extractor(file):
    """Extract text from a PDF file."""
    text = ""
    try:
        doc = fitz.open(stream=file.read(), filetype="pdf")
        for page in doc:
            text += page.get_text()
        doc.close()
    except Exception as e:
        logging.error(f"Failed to extract from PDF: {e}")
        text = "Failed to extract text due to an error."
    return text


def docx_extractor(file):
    """Extract text from a DOCX file."""
    text = ""
    try:
        doc = docx.Document(file)
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        logging.error(f"Failed to extract from DOCX: {e}")
        text = "Failed to extract text due to an error."
    return text.strip()



def weight_keywords(keywords, critical_terms):
    weighted_keywords = Counter()
    for keyword, count in keywords.items():
        # Increase weight for critical terms identified
        if keyword in critical_terms:
            # Adjust multiplier based on importance
            weighted_keywords[keyword] = count * 3
        else:
            weighted_keywords[keyword] = count
    return weighted_keywords


def identify_critical_terms(job_doc, feedback_terms=None):
    """Identifies critical terms from a pre-processed spaCy document object."""
    critical_terms = set()

    # Use existing spaCy document to identify terms without reprocessing
    # Ensure job_doc is already a spaCy Doc object and not a string
    phrase_matcher = PhraseMatcher(nlp.vocab)
    tech_skills = ['machine learning', 'deep learning', 'data science', 'artificial intelligence', 'natural language processing',
                   'computer vision', 'big data', 'cloud computing', 'software development', 'web development', 'database management',
                   'cybersecurity', 'networking', 'algorithm design']
    tech_patterns = [nlp.make_doc(text) for text in tech_skills]
    phrase_matcher.add("TECH_SKILLS", None, *tech_patterns)

    soft_skills = ['communication', 'problem-solving', 'teamwork', 'leadership', 'time management', 'adaptability',
                   'creativity', 'critical thinking', 'emotional intelligence', 'interpersonal skills', 'decision making', 'conflict resolution']
    soft_patterns = [nlp.make_doc(text) for text in soft_skills]
    phrase_matcher.add("SOFT_SKILLS", None, *soft_patterns)

    # Iterate over matches
    matches = phrase_matcher(job_doc)
    for match_id, start, end in matches:
        span = job_doc[start:end]  # The matched span
        critical_terms.add(span.text.lower())

    # Additional checks and term extraction from job_doc
    for token in job_doc:
        if token.dep_ in ["nsubj"] and token.head.lemma_ in ["require", "need", "include"]:
            critical_terms.add(token.lemma_.lower())

    if feedback_terms:
        critical_terms.update(feedback_terms)

    return critical_terms


def extract_keywords(doc):
    keywords = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct and token.pos_ in [
        'NOUN', 'VERB', 'PROPN']]
    # Include named entities as keywords, ensuring no duplication
    entities = [ent.lemma_.lower() for ent in doc.ents]
    all_keywords = keywords + list(set(entities) - set(keywords))
    return Counter(all_keywords)


def calculate_match_percentage(resume_keywords, weighted_job_keywords):
    """Calculate match percentage based on weighted keywords."""
    match_score = 0
    for keyword, weight in resume_keywords.items():
        if keyword in weighted_job_keywords:
            match_score += weight * weighted_job_keywords[keyword]
    total_possible_score = sum(weighted_job_keywords.values())
    match_percentage = (match_score / total_possible_score) * \
        100 if total_possible_score > 0 else 0
    return match_percentage

# compare_texts_char_by_char function is not defined in the model.py file. You can define it as follows:


def compare_texts_char_by_char(text1, text2):
    """Compare two texts character by character."""
    # Ensure both inputs are strings
    if not isinstance(text1, str) or not isinstance(text2, str):
        return {"error": "Both inputs should be strings."}

    # Split the texts into characters
    chars1, chars2 = list(text1), list(text2)

    # Calculate the number of matching characters
    matching_chars = sum(char1 == char2 for char1,
                         char2 in zip(chars1, chars2))

    # Calculate the percentage of matching characters
    match_percentage = (matching_chars / max(len(chars1), len(chars2))) * 100

    return {
        "match_percentage": match_percentage,
        "total_matching_chars": matching_chars
    }


def analyze_resume(structured_resume, job_description):
    """Analyze the complete resume against the job description."""
    # Safety checks for data quality
    if not structured_resume or not job_description:
        return {"error": "Missing resume or job_description."}

    # Process texts with NLP
    resume_doc = nlp(structured_resume)
    # Ensure this is only called once and handled properly
    job_doc = nlp(job_description)

    # Identify critical terms and extract & weight keywords
    # Pass the spaCy document directly, ensure no reprocessing
    critical_terms = identify_critical_terms(job_doc)
    resume_keywords = extract_keywords(resume_doc)
    job_keywords = extract_keywords(job_doc)
    # Copy to avoid modifying the original
    weighted_job_keywords = weight_keywords(
        job_keywords.copy(), critical_terms)

    # Calculate the match percentage
    match_percentage = calculate_match_percentage(
        resume_keywords, weighted_job_keywords)

    return {
        "match_percentage": match_percentage,
        "common_keywords": dict(resume_keywords & weighted_job_keywords),
        "resume_keywords": dict(resume_keywords),
        "job_keywords": dict(weighted_job_keywords),
        "critical_terms": list(critical_terms)
    }


def clean_and_structure_text(text):
    # Basic preprocessing to remove extra spaces and convert to lowercase
    text = ' '.join(text.split()).lower()

    # Patterns for extracting information
    patterns = {
        "name": r"^([a-z ,.'-]+)",  # Assuming name is at the start
        "email": r"([a-zA-Z0-9+_.-]+@[a-zA-Z0-9.-]+)",
        # US-format phone numbers
        "phone": r"(\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})",
        "links": r"(https?:\/\/[^\s]+)",  # URLs
        # Add more patterns as needed
    }

    structured_data = {}

    for category, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            structured_data[category] = match.group()

    # consider using simple string searches or more advanced NLP techniques like named entity recognition (NER)
    # you can use a custom function that takes the text and returns the structured data

    # Here's a simplistic approach:
    sections = ["education", "technical skills",
                "professional experience", "academic projects", "certifications"]
    for section in sections:
        start = text.find(section)
        if start != -1:
            end = len(text)
            for other in sections:
                temp_end = text.find(other, start + len(section))
                if 0 < temp_end < end:
                    end = temp_end
            structured_data[section] = text[start+len(section):end].strip()

    return structured_data


def clean_job_description(job_description, remove_stop_words=True):
    """
    Cleans the job description text by lowercasing, removing punctuation,
    and optionally removing stop words.

    Parameters:
    - job_description: The raw job description text.
    - remove_stop_words: Boolean flag to remove stop words.

    Returns:
    - Cleaned job description text.
    """
    # Remove non-alphanumeric characters (excluding spaces)
    job_description = re.sub(r"[^\w\s]", "", job_description)

    # Lowercase
    job_description = job_description.lower()

    if remove_stop_words:
        # Tokenize using spaCy and remove stop words
        doc = nlp(job_description)
        job_description = " ".join(
            [token.lemma_ for token in doc if not token.is_stop])
    else:
        # Tokenize using spaCy without removing stop words
        doc = nlp(job_description)
        job_description = " ".join([token.lemma_ for token in doc])
    return job_description


def extract_skills(text):
    """
    Extract skills from the given text. Skills are identified by specific keywords
    related to technical, soft, and hard skills.
    """
    # Define skills and their categories
    skill_keywords = {
        'technical_skills': [
            'python', 'java', 'c++', 'sql', 'tensorflow', 'pytorch', 'javascript', 'react', 'angular', 'node.js',
            'django', 'flask', 'ruby on rails', 'php', 'swift', 'kotlin', 'android development', 'ios development',
            'docker', 'kubernetes', 'aws', 'azure', 'google cloud', 'git', 'machine learning', 'deep learning',
            'data analysis', 'data visualization', 'big data', 'hadoop', 'spark', 'nosql', 'firebase', 'rest api',
            'graphql', 'sass', 'less', 'webpack', 'maven', 'jenkins', 'ci/cd', 'unity', 'unreal engine',
            'cybersecurity', 'penetration testing', 'vulnerability assessment', 'network security', 'blockchain', 'ethereum',
            'solidity', 'web3.js', 'truffle', 'ganache', 'cryptocurrency', 'smart contracts', 'decentralized applications',
            'computer vision', 'image processing', 'opencv', 'object detection', 'natural language processing', 'nlp',
            'chatbots', 'speech recognition', 'sentiment analysis', 'named entity recognition', 'text classification',
            'web development', 'frontend', 'backend', 'full stack', 'html', 'css', 'bootstrap', 'materialize', 'bulma',
            'jquery', 'ajax', 'laravel', 'codeigniter', 'symfony', 'express.js', 'mongodb', 'mysql', 'postgresql', 'sqlite',
            'oracle', 'graphql', 'socket.io', 'websockets', 'progressive web apps', 'pwa', 'single page applications',
            'spa', 'wordpress', 'shopify', 'magento', 'woocommerce', 'firebase', 'realtime database', 'cloud firestore',
            'authentication', 'authorization', 'oauth2', 'version control', 'api design', 'microservices', 'serverless architecture',
            'containerization', 'agile development', 'scrum', 'test-driven development', 'tdd', 'behavior-driven development',
            'bdd', 'pair programming', 'devops', 'site reliability engineering', 'sre', 'linux administration', 'bash scripting',
            'powershell', 'networking protocols', 'http', 'tcp/ip', 'sdlc', 'software development life cycle', 'user experience',
            'ux design', 'user interface', 'ui design', 'accessibility', 'seo', 'search engine optimization', 'performance optimization',
            'content management systems', 'cms', 'erp systems', 'enterprise resource planning', 'crm systems', 'customer relationship management',
            'cloud services', 'iaas', 'paas', 'saas', 'infrastructure as code', 'iac', 'ansible', 'terraform', 'cloudformation',
            'data engineering', 'etl', 'extract transform load', 'data warehousing', 'data lakes', 'apache airflow', 'apache beam',
            'apache kafka', 'stream processing', 'event-driven architecture', 'functional programming', 'reactive programming',
            'domain-driven design', 'ddd', 'quantitative analysis', 'data science', 'statistics', 'mathematical modeling',
            'predictive analytics', 'machine vision', 'augmented reality', 'ar', 'virtual reality', 'vr', 'mixed reality', 'mr',
            'iot', 'internet of things', 'robotics', 'automation', 'embedded systems', 'firmware development', 'hardware interfacing',
            '3d printing', 'additive manufacturing', 'scada', 'supervisory control and data acquisition', 'plc programming', 'industrial control systems',
            'quantum computing', 'genetic algorithms', 'simulation', 'game development', 'game design', 'digital signal processing', 'dsp',
            'audio processing', 'video processing', 'biometrics', 'facial recognition', 'fingerprint identification', 'cyber-physical systems',
            'bioinformatics', 'computational biology', 'health informatics', 'e-commerce', 'payment gateway integration', 'shopping cart development',
            'electronic design automation', 'eda', 'vlsi design', 'integrated circuit design', 'pcb design', 'printed circuit board design',
            'geospatial analysis', 'gis systems', 'geographic information systems', 'remote sensing', 'climate modeling', 'environmental modeling',
            '3d modeling', 'animation', 'rendering', 'graphic design', 'visual arts', 'artificial intelligence', 'ai', 'business intelligence',
            'bi', 'user research', 'customer insights', 'market analysis', 'competitive analysis', 'product management', 'project management',
            'resource management', 'risk management', 'strategic planning', 'financial modeling', 'budgeting', 'forecasting',
            'lean manufacturing', 'supply chain management', 'logistics', 'inventory management', 'quality control', 'six sigma',
            'continuous improvement', 'kaizen', 'iso standards', 'compliance', 'regulatory affairs', 'legal technology', 'edtech',
            'education technology', 'legaltech', 'fintech', 'financial technology', 'regtech', 'regulatory technology', 'healthtech',
            'medical technology', 'agritech', 'agricultural technology', 'cleantech', 'environmental technology', 'energy management',
            'smart grid technology', 'building automation', 'smart home technology', 'autonomous vehicles', 'self-driving cars',
            'drones', 'unmanned aerial vehicles', 'uv mapping', 'motion capture', 'mocap', 'vfx', 'visual effects', 'sfx', 'special effects',
            'mixed media', 'interactive media', 'digital marketing', 'social media management', 'influencer marketing', 'content strategy',
            'branding', 'public relations', 'pr', 'advertising', 'media planning', 'campaign management', 'event planning', 'event management',
            'community engagement', 'community management', 'customer service', 'technical support', 'it infrastructure', 'network administration',
            'system administration', 'database administration', 'cloud computing', 'virtualization', 'information security', 'cybersecurity',
            'threat intelligence', 'penetration testing', 'ethical hacking', 'security audits', 'incident response', 'forensics', 'data recovery',
            'privacy', 'data protection', 'gdpr', 'general data protection regulation', 'hipaa', 'health insurance portability and accountability act',
            'pci dss', 'payment card industry data security standard', 'compliance', 'audit', 'governance', 'policy development', 'change management',
            'organizational development', 'employee engagement', 'talent management', 'recruitment', 'talent acquisition', 'performance management',
            'compensation and benefits', 'payroll', 'employee relations', 'labor relations', 'industrial relations', 'training and development',
            'learning and development', 'succession planning', 'diversity and inclusion', 'culture development', 'organizational behavior',
            'leadership development', 'executive coaching', 'business coaching', 'mentoring', 'negotiation', 'conflict resolution', 'mediation',
            'arbitration', 'litigation support', 'legal research', 'legal writing', 'contract management', 'intellectual property', 'patents',
            'trademarks', 'copyright', 'licensing', 'compliance', 'corporate law', 'commercial law', 'employment law', 'labor law',
            'immigration law', 'family law', 'criminal law', 'civil law', 'tax law', 'international law', 'environmental law', 'energy law',
            'healthcare law', 'real estate law', 'personal injury law', 'bankruptcy law', 'financial law', 'securities law', 'compliance',
            'regulatory affairs', 'public policy', 'government relations', 'lobbying', 'advocacy', 'legislative analysis', 'policy analysis',
            'economic analysis', 'statistical analysis', 'data mining', 'machine learning', 'deep learning', 'artificial intelligence', 'ai',
            'computer vision', 'natural language processing', 'robotics', 'quantum computing', 'blockchain', 'cryptocurrency', 'big data', 'data science',
            'cybersecurity', 'information security', 'network security', 'cloud security', 'application security', 'data privacy', 'risk management',
            'business continuity', 'disaster recovery', 'emergency management', 'crisis management', 'operational resilience', 'physical security',
            'security operations', 'surveillance', 'investigations', 'corporate security', 'executive protection', 'asset protection', 'fraud prevention',
            'loss prevention', 'cyber forensics', 'digital forensics', 'incident response', 'threat hunting', 'vulnerability management', 'patch management',
            'identity and access management', 'iam', 'authentication', 'authorization', 'security architecture', 'security engineering', 'security testing',
            'penetration testing', 'red teaming', 'blue teaming', 'purple teaming', 'security operations center', 'soc', 'threat intelligence',
            'security awareness', 'security training', 'security policy', 'security governance', 'security compliance', 'security audit', 'security assessment',
            'security certification', 'security accreditation', 'security clearance', 'information assurance', 'data governance', 'data management',
            'data architecture', 'data modeling', 'database design', 'database development', 'database administration', 'data warehousing', 'data lake',
            'data integration', 'data migration', 'data quality', 'data security', 'data privacy', 'data protection', 'data retention', 'data disposal',
            'data analytics', 'business intelligence', 'bi', 'reporting', 'dashboards', 'visualization', 'analytics', 'predictive analytics', 'prescriptive analytics',
            'descriptive analytics', 'diagnostic analytics', 'statistical analysis'
        ],
        'soft_skills': [
            'teamwork', 'communication', 'leadership', 'problem-solving', 'time management', 'critical thinking',
            'emotional intelligence', 'creativity', 'adaptability', 'work ethic', 'interpersonal skills',
            'decision making', 'stress management', 'conflict resolution', 'negotiation', 'persuasion',
            'project management', 'motivation', 'patience', 'empathy', 'cultural awareness', 'active listening',
            'public speaking', 'presentation skills', 'attention to detail', 'organization', 'collaboration',
            'customer service', 'sales', 'marketing', 'business development', 'account management', 'client relations',
            'relationship management', 'networking', 'influencing', 'mentoring', 'coaching', 'training',
        ],
        'hard_skills': [
            'project management', 'budgeting', 'financial forecasting', 'excel', 'power bi', 'tableau',
            'salesforce', 'marketing automation', 'seo/sem', 'content creation', 'graphic design',
            'user interface design', 'user experience design', 'typography', '3d modeling', 'video editing',
            'translation', 'copywriting', 'statistical analysis', 'electrical engineering', 'mechanical engineering',
            'educational curriculum design', 'legal research', 'medical coding', 'pharmaceutical research',
            'construction management', 'real estate analysis', 'property management', 'logistics planning',
        ]
    }

    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    for category, keywords in skill_keywords.items():
        # Create a Doc object for each keyword
        patterns = [nlp.make_doc(keyword) for keyword in keywords]
        matcher.add(category, patterns)  # Add patterns to the matcher

    doc = nlp(text.lower())  # Convert text to lowercase for matching
    matches = matcher(doc)
    skills_found = defaultdict(list)

    for match_id, start, end in matches:
        rule_id = nlp.vocab.strings[match_id]  # Get the category of the match
        span = doc[start:end]  # The matched span
        # Append the skill to the corresponding category
        skills_found[rule_id].append(span.text)
        # Remove duplicates from skills_found
        
    for category in skills_found:
        skills_found[category] = list(set(skills_found[category]))
    return skills_found


def remove_duplicates(skills_found):
    """Remove duplicates from the extracted skills."""
    for category in skills_found:
        skills_found[category] = list(set(skills_found[category]))
    return skills_found


def has_vector(token):
    """Check if token has a vector representation."""
    return token.has_vector and token.vector_norm


def get_missing_skills(resume_skills, job_skills):
    """
    Identify skills listed in the job description not found in the resume.
    Prioritize suggestions based on the frequency of skills in the job description.
    """
    missing_skills = defaultdict(list)
    for category, skills in job_skills.items():
        # Convert job skills to spaCy tokens
        job_skill_tokens = [nlp(skill)
                            for skill in skills if has_vector(nlp(skill))]
        # Make sure to check if resume skills are not empty and tokens have vectors
        resume_skill_tokens = [nlp(skill) for skill in resume_skills.get(
            category, []) if has_vector(nlp(skill))]
        for job_skill_token in job_skill_tokens:
            if not has_vector(job_skill_token):
                continue
            if not any(job_skill_token.similarity(resume_skill_token) > 2 for resume_skill_token in resume_skill_tokens):
                missing_skills[category].append(job_skill_token.text)
    return missing_skills


def suggest_improvements(resume_text, job_description_text):
    # Extract skills from both texts
    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills(job_description_text)

    # Remove duplicates from extracted skills
    resume_skills = remove_duplicates(resume_skills)
    job_skills = remove_duplicates(job_skills)

    # Identify missing skills
    missing_skills = get_missing_skills(resume_skills, job_skills)

    # Prioritize missing skills based on their frequency in the job description
    job_description_counter = Counter(job_description_text.split())

    for category in missing_skills:
        missing_skills[category] = sorted(
            missing_skills[category], key=lambda skill: job_description_counter.get(skill, 0), reverse=True
        )

    # Format suggestions for output
    suggestions = []
    for category, skills in missing_skills.items():
        if skills:
            formatted_skills = ', '.join(skills)
            suggestions.append(f"Consider adding these {category.replace('_', ' ')}: {formatted_skills}")

    # Ensure this always returns a list
    return suggestions



# Set up the model configuration for text generation
generation_config = {
    "temperature": 0.4,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 4096,
}

# Define safety settings for content generation
safety_settings = [
    {"category": f"HARM_CATEGORY_{category}",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
    for category in ["HARASSMENT", "HATE_SPEECH", "SEXUALLY_EXPLICIT", "DANGEROUS_CONTENT"]
]


def generate_response_from_gemini(input_text, job_description):
    input_prompt = """
    As an experienced Applicant Tracking System (ATS) analyst,
    with profound knowledge in technology, software engineering, data science, software development, cybersecurity engineering, network engineering, 
    achine Learning engineering, data engineers, data analytics and big data engineering, your role involves evaluating resumes against job descriptions.
    Recognizing the competitive job market, provide top-notch assistance for resume improvement.
    Your goal is to analyze the resume against the given job description, 
    assign a percentage match based on key criteria, and pinpoint missing keywords accurately.
    Resume:{text}
    Description:{job_description}
    I want the response in one single string having the structure
    {{"Job Description Match":"%","Missing Keywords":"","Candidate Summary":"","Experience":"", "Missing Technical Skills":"", "Missing Soft Skills":"", "Missing Hard Skills":""\n}}
    """.format(text=input_text, job_description=job_description)

    # Create a GenerativeModel instance with 'gemini-pro' as the model type
    llm = genai.GenerativeModel(
        model_name="gemini-pro",
        generation_config=generation_config,
        safety_settings=safety_settings,
    )

    try:
        # Generate content based on the input text
        output = llm.generate_content(input_prompt)

        # Parse the result string into JSON format
        response_json = json.loads(output.text)
        print("Parsed Google Gemini Response: ", response_json)

        # Return the parsed JSON response
        return response_json

    except json.JSONDecodeError as e:
        print(f"Error parsing JSON from Gemini: {e}")
        return {
            "Job Description Match": "0%",
            "Missing Keywords": "N/A",
            "Candidate Summary": "Error generating candidate summary.",
            "Experience": "Error retrieving experience details.",
            "Missing Technical Skills": "N/A",
            "Missing Soft Skills": "N/A",
            "Missing Hard Skills": "N/A"
        }
    except Exception as e:
        print(f"Error generating response from Gemini: {e}")
        return {
            "Job Description Match": "0%",
            "Missing Keywords": "N/A",
            "Candidate Summary": "An unexpected error occurred.",
            "Experience": "Unable to retrieve experience.",
            "Missing Technical Skills": "N/A",
            "Missing Soft Skills": "N/A",
            "Missing Hard Skills": "N/A"
        }


def safely_parse_json(json_str):
    try:
        return json.loads(json_str)
    except json.decoder.JSONDecodeError:
        # Log error or take other actions as necessary
        return {}  # Return an empty dict if parsing fails


def combined_results(input_text, job_description):
    combined_results = {
        "Combined Match Percentage": "",
        "Unified Missing Keywords": [],
        "Improvement Suggestions": []
    }

    # Assuming `generate_response_from_gemini` returns a JSON string
    gemini_results = json.loads(
        generate_response_from_gemini(input_text, job_description))
    custom_results = suggest_improvements(
        input_text, job_description)  # This is a list

    # Directly use the Gemini match percentage if available
    if 'Job Description Match' in gemini_results:
        combined_results['Combined Match Percentage'] = float(
            gemini_results['Job Description Match'].strip('%'))

    # Combine missing keywords from Gemini results
    gemini_keywords = gemini_results.get('Missing Keywords', '').split(', ')
    # Since `custom_results` is a list, we directly use it without calling `.get`
    combined_results['Improvement Suggestions'] = custom_results + \
        gemini_keywords

    return combined_results


def identify_and_highlight_errors(text):
    tool = language_tool_python.LanguageTool('en-US')
    matches = tool.check(text)

    highlighted_text = text
    error_details = []  # To store detailed information about each error

    # For readability scores (ensure textstat is installed and imported)
    readability_scores = {
        "Flesch Reading Ease": textstat.flesch_reading_ease(text),
        "Flesch-Kincaid Grade Level": textstat.flesch_kincaid_grade(text)
    }

    # Sort matches in reverse to avoid messing up the indices
    matches.sort(key=lambda match: match.offset, reverse=True)

    for match in matches:
        start, end = match.offset, match.offset + match.errorLength
        original_text = text[start:end]  # Use original text here for clarity

        suggestions = [suggestion for suggestion in match.replacements]

        # Highlight errors with a tooltip for explanation and suggestions
        corrected_text = f'<span style="color: red; text-decoration: underline;" title="{match.message} Suggestions: {", ".join(suggestions)}">{original_text}</span>'
        highlighted_text = highlighted_text[:start] + \
            corrected_text + highlighted_text[end:]

        # Append error details
        error_details.append({
            "text": original_text,
            "type": match.ruleIssueType,
            "explanation": match.message,
            "suggestions": suggestions  # Including suggestions for corrections
        })

    return Markup(highlighted_text), error_details, readability_scores
