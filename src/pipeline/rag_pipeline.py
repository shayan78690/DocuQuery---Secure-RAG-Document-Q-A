# src/pipeline/rag_pipeline.py

import os
import multiprocessing
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers

# --- Configuration ---
MODEL_PATH = "llm_model/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

def load_llm():
    """
    Loads and returns the language model.
    """
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found at path: {MODEL_PATH}")
        print("Please ensure the model file is downloaded, correctly named, and placed in the 'llm_model' directory.")
        return None

    try:
        cpu_count = multiprocessing.cpu_count()
        threads_to_use = max(1, cpu_count // 2)
        print(f"Using {threads_to_use} CPU threads for the model.")

        llm = CTransformers(
            model=MODEL_PATH,
            model_type='llama',
            config={
                'max_new_tokens': 150,
                'temperature': 0.7,
                'threads': threads_to_use,
                'context_length': 2048
            }
        )
        return llm
    except Exception as e:
        print(f"An error occurred during model loading: {e}")
        return None

def create_vector_store(file_path, vector_store_path):
    """
    Processes a document and saves its vector store to a specified path.
    """
    try:
        file_extension = os.path.splitext(file_path)[1].lower()

        if file_extension == '.pdf':
            loader = PyMuPDFLoader(file_path)
        elif file_extension == '.docx':
            loader = Docx2txtLoader(file_path)
        elif file_extension == '.txt':
            loader = UnstructuredFileLoader(file_path)
        else:
            print(f"Unsupported file type: {file_extension}")
            return False
        
        print(f"Loading document with {type(loader).__name__}")
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
        text_chunks = text_splitter.split_documents(documents)
        embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
        
        db = FAISS.from_documents(text_chunks, embeddings)
        db.save_local(vector_store_path)
        
        return True
    except Exception as e:
        print(f"An error occurred during vector store creation: {e}")
        return False
