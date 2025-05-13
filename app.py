import os
import json
import tempfile
from flask import Flask, render_template, request, jsonify
import fitz  # PyMuPDF
import docx
import re
import requests
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ollama API configuration
OLLAMA_API_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "llama3.1:8b"  # Corrected model name format

def extract_text_from_pdf(file_path):
    """Extract text from PDF file."""
    text = ""
    try:
        with fitz.open(file_path) as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
    return text

def extract_text_from_docx(file_path):
    """Extract text from DOCX file."""
    text = ""
    try:
        doc = docx.Document(file_path)
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
    except Exception as e:
        print(f"Error extracting text from DOCX: {e}")
    return text

def extract_text_from_file(file_path):
    """Extract text from supported file types."""
    extension = os.path.splitext(file_path)[1].lower()
    
    if extension == '.pdf':
        return extract_text_from_pdf(file_path)
    elif extension == '.docx':
        return extract_text_from_docx(file_path)
    elif extension == '.txt':
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    else:
        return "Unsupported file format"

def generate_llm_prompt(resume_text):
    """Generate a structured prompt for the LLM."""
    return f"""
    You are a resume parsing expert. Extract structured information from the following resume text.
    Return a JSON object with the following structure:
    {{
        "personal_details": {{
            "name": "Full Name",
            "phone": "Phone Number",
            "email": "Email Address",
            "linkedin": "LinkedIn URL",
            "location": "Location/Address",
            "other_contacts": ["Any other contact information"]
        }},
        "about": "Extract the about/summary/profile section text (this might not have a header and could be at the beginning)",
        "sections": [
            {{
                "title": "Section Title (e.g. Experience, Education, Skills)",
                "content": "Full text content of this section"
            }},
            // More sections...
        ]
    }}
    
    When extracting sections, make sure to:
    1. Preserve the original section headers exactly as they appear
    2. Include ALL content under each section 
    3. Keep the order of sections as they appear in the resume
    4. Don't miss any sections, even minor ones
    5. Make sure the JSON is valid and properly formatted
    
    Resume text:
    {resume_text}
    """

def query_model(prompt):
    """Query the local Llama model via Ollama API."""
    try:
        # Correct payload structure for the /api/chat endpoint
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a resume parsing expert that extracts structured information accurately. Only respond with the JSON."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": False,
            "temperature": 0.1  # Low temperature for more deterministic parsing
        }
        
        response = requests.post(OLLAMA_API_URL, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            
            # Extract the generated text from the message response
            if 'message' in result and 'content' in result['message']:
                text = result['message']['content']
                
                # Find JSON part using regex
                json_match = re.search(r'```json\s*([\s\S]*?)\s*```', text)
                if json_match:
                    text = json_match.group(1)
                else:
                    # Try to find JSON with curly braces
                    json_match = re.search(r'({[\s\S]*})', text)
                    if json_match:
                        text = json_match.group(1)
                
                try:
                    # Try to parse as JSON
                    return json.loads(text)
                except json.JSONDecodeError:
                    # If parsing fails, return the raw text for debugging
                    return {"error": "Failed to parse JSON", "raw_response": text}
            
            return {"error": "No message field in result", "raw_result": result}
        else:
            return {"error": f"API request failed with status code {response.status_code}", "details": response.text}
    
    except Exception as e:
        return {"error": f"Exception while querying model: {str(e)}"}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/parse', methods=['POST'])
def parse_resume():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    
    # Check if the file extension is allowed
    allowed_extensions = {'.pdf', '.docx', '.txt'}
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in allowed_extensions:
        return jsonify({"error": f"File type not supported. Please upload a PDF, DOCX, or TXT file."})
    
    try:
        # Save the file temporarily
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Extract text from the file
        resume_text = extract_text_from_file(file_path)
        
        # Remove the temporary file
        os.remove(file_path)
        
        if not resume_text or resume_text == "Unsupported file format":
            return jsonify({"error": "Could not extract text from the file"})
        
        # Generate prompt and query LLM
        prompt = generate_llm_prompt(resume_text)
        parsed_data = query_model(prompt)
        
        return jsonify(parsed_data)
    
    except Exception as e:
        return jsonify({"error": f"Error processing file: {str(e)}"})

if __name__ == '__main__':
    app.run(debug=True)