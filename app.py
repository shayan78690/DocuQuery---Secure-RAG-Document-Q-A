# app.py

import os
import hashlib
from flask import Flask, render_template, request, Response
from werkzeug.utils import secure_filename
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.prompts import PromptTemplate
from src.prompt import prompt_template
from src.pipeline.rag_pipeline import create_vector_store, load_llm

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
PERSISTENT_STORAGE_PATH = "persistent_storage"
VECTOR_STORE_BASE_PATH = os.path.join(PERSISTENT_STORAGE_PATH, "vector_stores")
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(VECTOR_STORE_BASE_PATH, exist_ok=True)

print("Loading the language model... This may take a moment.")
llm = load_llm()
print("Model loaded successfully.")

def get_file_hash(file_stream):
    """Calculates the SHA256 hash of a file stream."""
    hash_sha256 = hashlib.sha256()
    file_stream.seek(0)
    for chunk in iter(lambda: file_stream.read(4096), b""):
        hash_sha256.update(chunk)
    file_stream.seek(0)
    return hash_sha256.hexdigest()

def allowed_file(filename):
    """Checks if the file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Renders the main page."""
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_document():
    if 'document' not in request.files:
        return Response("No file part in the request.", status=400)
    
    file = request.files['document']
    question = request.form.get('question')

    if file.filename == '' or not question:
        return Response("No file selected or no question asked.", status=400)

    if file and allowed_file(file.filename):
        filepath = "" # Define filepath to be accessible in finally block
        try:
            file_hash = get_file_hash(file.stream)
            vector_store_path = os.path.join(VECTOR_STORE_BASE_PATH, file_hash)

            if not os.path.exists(vector_store_path):
                print(f"Vector store for hash {file_hash} not found. Creating new one.")
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
                file.save(filepath)
                
                if not create_vector_store(filepath, vector_store_path):
                    return Response("Failed to process the document.", status=500)
                
                if os.path.exists(filepath):
                    os.remove(filepath)
            else:
                print(f"Found existing vector store for hash {file_hash}.")

            if not llm:
                 return Response("The Language Model is not available.", status=500)
            
            embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
            vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
            
            retriever = vector_store.as_retriever(search_kwargs={'k': 2})
            docs = retriever.get_relevant_documents(question)
            context = "\n\n".join([doc.page_content for doc in docs])

            prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
            formatted_prompt = prompt.format(context=context, question=question)
            
            def generate():
                for token in llm.stream(formatted_prompt):
                    yield token

            return Response(generate(), mimetype='text/plain')

        except Exception as e:
            print(f"Error during processing: {e}")
            if filepath and os.path.exists(filepath):
                os.remove(filepath)
            return Response(f"An unexpected error occurred: {str(e)}", status=500)

    return Response("Invalid file type.", status=400)

if __name__ == '__main__':
    app.run(debug=False)
