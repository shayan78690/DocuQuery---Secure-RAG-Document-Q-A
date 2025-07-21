# DocuQuery: Local Document Q&A with RAG

DocuQuery is a full-stack web application that allows you to ask questions about your own documents (`.pdf`, `.docx`, `.txt`) and get answers from a locally-run AI model. It uses a Retrieval-Augmented Generation (RAG) pipeline to ensure answers are based *only* on the content of your uploaded file, guaranteeing data privacy.



### Key Features

-   **Private & Secure:** Your documents are processed locally and never sent to external services.
-   **Multi-File Support:** Upload and query PDF, Microsoft Word, and text files.
-   **Streaming Answers:** The AI's response is streamed in real-time for an improved user experience.
-   **Efficient Caching:** Documents are processed only once and their vector stores are cached for instant re-use.
-   **Containerized:** Fully containerized with Docker for easy deployment and scalability.

### Technology Stack

-   **Backend:** Python, Flask, LangChain, Gunicorn
-   **AI Model:** TinyLlama (running locally via CTransformers)
-   **Vector DB/Search:** FAISS (Facebook AI Similarity Search)
-   **Frontend:** HTML, Tailwind CSS, Vanilla JavaScript
-- **Deployment:** Docker, Render

### How to Run Locally


1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/docuquery.git](https://github.com/your-username/docuquery.git)
    cd docuquery
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Download the LLM:**
    -   Create a folder named `llm_model`.
    -   Download the [TinyLlama GGUF model](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf) and place it inside `llm_model`.
5.  **Run the application:**
    ```bash
    python app.py
    ```
