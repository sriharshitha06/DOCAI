import json
import os
import re
import pdfplumber
from PIL import Image, ImageEnhance
import pytesseract
import cv2
import numpy as np
from colorthief import ColorThief
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
from flask import Flask, render_template, jsonify, request, url_for
from flask_wtf.csrf import CSRFProtect
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from dotenv import load_dotenv
from datetime import datetime
import logging
import sqlite3
from contextlib import contextmanager
import time
from celery_config import make_celery
from src.prompt import system_prompt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_APP_KEY', os.urandom(24))
app.config['SESSION_COOKIE_SECURE'] = True
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB file size limit
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching for static files during development

# Initialize CSRF protection
csrf = CSRFProtect(app)

# Initialize Celery
celery = make_celery(app)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

load_dotenv()
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    logging.error("PINECONE_API_KEY is not set in environment variables")
    raise ValueError("PINECONE_API_KEY is required")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Load vector embeddings
embeddings = download_hugging_face_embeddings()
index_name = "docai"

def get_pinecone_vector_store(namespace='default'):
    try:
        return PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=embeddings,
            namespace=namespace
        )
    except Exception as e:
        logging.error(f"Failed to initialize Pinecone index for namespace {namespace}: {e}")
        raise

def get_rag_chain(namespace):
    try:
        docsearch = get_pinecone_vector_store(namespace)
        retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        llm = OllamaLLM(model="tinyllama")
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{context}\n{input}\nQuestion: {question}")
        ])
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        return create_retrieval_chain(retriever, question_answer_chain)
    except Exception as e:
        logging.error(f"Failed to create RAG chain for namespace {namespace}: {e}")
        raise

# File upload configuration
UPLOAD_FOLDER = 'Uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'pdf', 'txt', 'jpg', 'jpeg', 'png'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# SQLite database initialization
def init_db():
    with get_db() as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                user_id TEXT,
                input TEXT,
                response TEXT,
                type TEXT,
                FOREIGN KEY (user_id) REFERENCES users(email)
            )
        ''')
        conn.commit()

@contextmanager
def get_db():
    conn = sqlite3.connect('history.db')
    try:
        yield conn
    finally:
        conn.close()

def load_user_history(user_id):
    with get_db() as conn:
        cursor = conn.execute('SELECT timestamp, user_id, input, response, type FROM history WHERE user_id = ? ORDER BY timestamp DESC', (user_id,))
        return [
            {'timestamp': row[0], 'user_id': row[1], 'input': row[2], 'response': row[3], 'type': row[4]}
            for row in cursor.fetchall()
        ]

def save_history(history):
    with get_db() as conn:
        for entry in history:
            conn.execute(
                'INSERT INTO history (timestamp, user_id, input, response, type) VALUES (?, ?, ?, ?, ?)',
                (entry['timestamp'], entry.get('user_id'), entry['input'], entry['response'], entry['type'])
            )
        conn.commit()

def clear_user_history(user_id):
    with get_db() as conn:
        conn.execute('DELETE FROM history WHERE user_id = ?', (user_id,))
        conn.commit()
    # Clear Pinecone namespace
    try:
        docsearch = get_pinecone_vector_store(namespace=user_id)
        docsearch.delete(delete_all=True, namespace=user_id)
        logging.info(f"Cleared Pinecone namespace for user {user_id}")
    except Exception as e:
        logging.error(f"Failed to clear Pinecone namespace for user {user_id}: {e}")

# Initialize database
init_db()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(pdf_path):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ''
            for page in pdf.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text + '\n'
            if not text.strip():
                logging.warning(f"No text extracted from PDF: {pdf_path}")
            return text
    except Exception as e:
        logging.error(f"Error reading PDF {pdf_path}: {e}")
        return ""

def preprocess_image_for_ocr(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            logging.error(f"Failed to load image: {image_path}")
            return None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        blurred = cv2.GaussianBlur(thresh, (5, 5), 0)
        processed_image = Image.fromarray(blurred)
        return processed_image
    except Exception as e:
        logging.error(f"Error preprocessing image {image_path}: {e}")
        return None

def analyze_image(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return "Unable to load image for analysis."
        height, width, _ = img.shape
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APROX_SIMPLE)
        num_shapes = len(contours)
        color_thief = ColorThief(image_path)
        dominant_color = color_thief.get_color(quality=1)
        description = (
            f"The image is a {width}x{height} pixel image. "
            f"It contains {num_shapes} distinct regions or shapes detected via edge analysis. "
            f"The dominant color is RGB{dominant_color}."
        )
        return description
    except Exception as e:
        logging.error(f"Error analyzing image {image_path}: {e}")
        return "Unable to analyze image content due to an error."

def extract_text_from_image(image_path):
    try:
        processed_image = preprocess_image_for_ocr(image_path)
        if not processed_image:
            raise Exception("Image preprocessing failed")
        psm_modes = [6, 3, 11]
        extracted_text = ""
        for psm in psm_modes:
            text = pytesseract.image_to_string(processed_image, config=f"--psm {psm}")
            if text.strip():
                extracted_text = text.strip()
                break
        if not extracted_text:
            description = analyze_image(image_path)
            logging.warning(f"No text extracted from image: {image_path}. Providing image description: {description}")
            return "", description
        return extracted_text, None
    except Exception as e:
        logging.error(f"Error extracting text from image {image_path}: {e}")
        description = analyze_image(image_path)
        logging.warning(f"OCR failed. Providing image description: {description}")
        return "", description

def clean_text(input_text):
    try:
        if not input_text or not isinstance(input_text, str):
            raise ValueError("Invalid input text. It must be a non-empty string.")
        patterns = [
            (r"System:.*?Context:.*?(?=\n|$)", ""),
            (r"(GALE ENCYCLOPEDIA OF MEDICINE|Researchers,\s*Inc\..*?)(?=\n|$)", ""),
            (r"Orale isotr[eé]tinoin", "Oral isotretinoin"),
            (r"tr[eé]ti[nñ]oin", "tretinoin"),
            (r"/C15", ""),
            (r'\s+', ' ')
        ]
        for pattern, replacement in patterns:
            input_text = re.sub(pattern, replacement, input_text, flags=re.DOTALL | re.IGNORECASE)
        return input_text.strip()
    except ValueError as e:
        logging.error(f"Error during text cleaning: {e}")
        return ""
    except Exception as e:
        logging.error(f"Unexpected error during text cleaning: {e}")
        return input_text if isinstance(input_text, str) else ""

class User(UserMixin):
    def __init__(self, email):
        self.id = email
        self.email = email

@login_manager.user_loader
def load_user(email):
    with get_db() as conn:
        cursor = conn.execute('SELECT email FROM users WHERE email = ?', (email,))
        user = cursor.fetchone()
        if user:
            return User(user[0])
        return None

@celery.task
def store_documents_in_pinecone(report_text, summary, user_id):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    report_doc = Document(
        page_content=report_text,
        metadata={"user_id": user_id, "type": "report", "timestamp": timestamp}
    )
    summary_doc = Document(
        page_content=summary,
        metadata={"user_id": user_id, "type": "summary", "timestamp": timestamp}
    )
    try:
        docsearch = get_pinecone_vector_store(namespace=user_id)
        docsearch.add_documents([report_doc, summary_doc], namespace=user_id)
        logging.info(f"Stored documents in Pinecone for user {user_id}")
    except Exception as e:
        logging.error(f"Error storing documents in Pinecone for user {user_id}: {e}")

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/chat")
@login_required
def chat():
    return render_template('chat.html', user_id=current_user.email)

@app.route("/history")
@login_required
def view_history():
    history = load_user_history(current_user.email)
    return render_template("history.html", history=history)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        if not email or not password:
            return jsonify({'error': 'Email and password are required'}), 400
        password_hash = generate_password_hash(password, method='pbkdf2:sha256')
        try:
            with get_db() as conn:
                conn.execute('INSERT INTO users (email, password_hash, created_at) VALUES (?, ?, ?)',
                             (email, password_hash, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
                conn.commit()
            user = User(email)
            login_user(user)
            return jsonify({'message': 'Registration successful', 'redirect': url_for('index')}), 200
        except sqlite3.IntegrityError:
            return jsonify({'error': 'Email already exists'}), 400
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        with get_db() as conn:
            cursor = conn.execute('SELECT email, password_hash FROM users WHERE email = ?', (email,))
            user_data = cursor.fetchone()
        if user_data and check_password_hash(user_data[1], password):
            user = User(user_data[0])
            login_user(user)
            return jsonify({'message': 'Login successful', 'redirect': url_for('index')}), 200
        return jsonify({'error': 'Invalid credentials'}), 401
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return jsonify({'message': 'Logged out successfully', 'redirect': url_for('index')}), 200

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if not request.form.get('prompt'):
        return jsonify({"error": "Prompt is required"}), 400

    prompt_input = request.form['prompt']
    prompt_input = clean_text(prompt_input)

    user_id = current_user.email
    rag_chain = get_rag_chain(user_id)

    with get_db() as conn:
        cursor = conn.execute('SELECT input FROM history WHERE user_id = ? AND type = ? ORDER BY timestamp DESC LIMIT 50',
                             (user_id, 'query'))
        history = [row[0] for row in cursor.fetchall()]

    history.append(prompt_input)
    context = "\n".join(history)
    question = prompt_input

    try:
        response = rag_chain.invoke({
            "context": context,
            "input": prompt_input,
            "question": question
        })
    except Exception as e:
        logging.error(f"Error in RAG chain for user {user_id}: {e}")
        return jsonify({"error": "Failed to process query"}), 500

    history_entry = {
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "user_id": user_id,
        "input": prompt_input,
        "response": response['answer'],
        "type": "query"
    }
    save_history([history_entry])

    return jsonify({"response": response['answer']})

@app.route('/upload', methods=['POST'])
@login_required
def upload_report():
    start_time = time.time()
    if 'file' not in request.files:
        return jsonify({'error': 'Missing file'}), 400

    file = request.files['file']
    user_id = current_user.email

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        logging.info(f"File upload took {time.time() - start_time:.4f} seconds")

        start_extract = time.time()
        if filename.endswith('.pdf'):
            report_text = extract_text_from_pdf(file_path)
            description = None
        elif filename.endswith(('.jpg', '.jpeg', '.png')):
            report_text, description = extract_text_from_image(file_path)
        else:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    report_text = f.read()
                description = None
            except UnicodeDecodeError:
                logging.error(f"Failed to decode text file: {file_path}")
                return jsonify({'error': 'Unsupported file encoding'}), 400
        logging.info(f"Text extraction took {time.time() - start_extract:.4f} seconds")

        if not report_text and description:
            history = load_user_history(user_id)
            history.append({
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "user_id": user_id,
                "input": "Uploaded medical report",
                "response": description,
                "type": "report_upload"
            })
            save_history([history[-1]])
            logging.info(f"Total upload processing took {time.time() - start_time:.4f} seconds")
            return jsonify({
                'message': 'Image analyzed successfully, but no text was extracted for summarization.',
                'description': description
            })

        if not report_text:
            return jsonify({'error': 'No text extracted from file'}), 400

        start_clean = time.time()
        report_text = clean_text(report_text)
        logging.info(f"Text cleaning took {time.time() - start_clean:.4f} seconds")

        start_summary = time.time()
        rag_chain = get_rag_chain(user_id)
        summary_prompt = f"Summarize the following medical report in 3-5 sentences, focusing on key information such as diagnosis, treatment, medications, and test results:\n{report_text}"
        try:
            summary_response = rag_chain.invoke({
                "context": report_text,
                "input": summary_prompt,
                "question": "Provide a concise summary of the medical report."
            })
            summary = summary_response['answer']
        except Exception as e:
            logging.error(f"Error generating summary for user {user_id}: {e}")
            return jsonify({'error': 'Failed to generate summary'}), 500
        logging.info(f"Summary generation took {time.time() - start_summary:.4f} seconds")

        start_store = time.time()
        try:
            store_documents_in_pinecone.delay(report_text, summary, user_id)
        except Exception as e:
            logging.error(f"Failed to queue Pinecone storage task for user {user_id}: {e}")
        logging.info(f"Queueing Pinecone storage took {time.time() - start_store:.4f} seconds")

        start_history = time.time()
        history = load_user_history(user_id)
        history.append({
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "user_id": user_id,
            "input": "Uploaded medical report",
            "response": summary,
            "type": "report_upload"
        })
        save_history([history[-1]])
        logging.info(f"History update took {time.time() - start_history:.4f} seconds")

        logging.info(f"Total upload processing took {time.time() - start_time:.4f} seconds")
        return jsonify({
            'message': 'Report processed successfully',
            'summary': summary
        })
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/analyze_scan', methods=['POST'])
@login_required
def analyze_scan():
    start_time = time.time()
    if 'file' not in request.files:
        return jsonify({'error': 'Missing file'}), 400

    file = request.files['file']
    user_id = current_user.email

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename) and file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        logging.info(f"Scan upload took {time.time() - start_time:.4f} seconds")

        start_extract = time.time()
        scan_text, description = extract_text_from_image(file_path)
        logging.info(f"Text extraction took {time.time() - start_extract:.4f} seconds")

        if not scan_text and description:
            history = load_user_history(user_id)
            history.append({
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "user_id": user_id,
                "input": "Uploaded medical scan",
                "response": description,
                "type": "scan_analysis"
            })
            save_history([history[-1]])
            logging.info(f"Total scan processing took {time.time() - start_time:.4f} seconds")
            return jsonify({
                'message': 'Scan analyzed successfully, but no text was extracted.',
                'analysis': description
            })

        content = scan_text if scan_text else description
        if not content:
            return jsonify({'error': 'No content extracted from scan'}), 400

        start_clean = time.time()
        content = clean_text(content)
        logging.info(f"Text cleaning took {time.time() - start_clean:.4f} seconds")

        start_analysis = time.time()
        rag_chain = get_rag_chain(user_id)
        analysis_prompt = f"Analyze the following medical scan description or extracted text and provide a detailed medical interpretation, including potential diagnoses, observations, and recommendations:\n{content}"
        try:
            analysis_response = rag_chain.invoke({
                "context": content,
                "input": analysis_prompt,
                "question": "Provide a detailed medical analysis of the scan."
            })
            analysis = analysis_response['answer']
        except Exception as e:
            logging.error(f"Error generating scan analysis for user {user_id}: {e}")
            return jsonify({'error': 'Failed to generate scan analysis'}), 500
        logging.info(f"Scan analysis took {time.time() - start_analysis:.4f} seconds")

        start_store = time.time()
        try:
            store_documents_in_pinecone.delay(content, analysis, user_id)
        except Exception as e:
            logging.error(f"Failed to queue Pinecone storage task for user {user_id}: {e}")
        logging.info(f"Queueing Pinecone storage took {time.time() - start_store:.4f} seconds")

        start_history = time.time()
        history = load_user_history(user_id)
        history.append({
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "user_id": user_id,
            "input": "Uploaded medical scan",
            "response": analysis,
            "type": "scan_analysis"
        })
        save_history([history[-1]])
        logging.info(f"History update took {time.time() - start_history:.4f} seconds")

        logging.info(f"Total scan processing took {time.time() - start_time:.4f} seconds")
        return jsonify({
            'message': 'Scan analyzed successfully',
            'analysis': analysis
        })
    return jsonify({'error': 'Invalid file type. Only images (jpg, jpeg, png) are allowed.'}), 400

@app.route('/history/<user_id>', methods=['GET'])
@login_required
def get_history(user_id):
    if user_id != current_user.email:
        return jsonify({'error': 'Unauthorized access'}), 403
    history = load_user_history(user_id)
    return jsonify(history)

@app.route('/history/<user_id>/clear', methods=['POST'])
@login_required
def clear_history(user_id):
    if user_id != current_user.email:
        return jsonify({'error': 'Unauthorized access'}), 403
    try:
        clear_user_history(user_id)
        return jsonify({'message': 'History cleared successfully'}), 200
    except Exception as e:
        logging.error(f"Error clearing history for user {user_id}: {e}")
        return jsonify({'error': 'Failed to clear history'}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)