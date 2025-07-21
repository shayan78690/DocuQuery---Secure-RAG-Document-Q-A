# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install curl to download the model
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy the dependencies file to the working directory
COPY requirements.txt .

# --- OPTIMIZATION: Install CPU-only PyTorch first ---
# This prevents the download of massive, unnecessary NVIDIA CUDA files.
RUN pip install torch --no-cache-dir --index-url https://download.pytorch.org/whl/cpu

# --- Install the rest of the dependencies ---
RUN pip install --no-cache-dir --use-deprecated=legacy-resolver -r requirements.txt

# Copy the entire project to the working directory
COPY . .

# Download the LLM model into the image during the build process
RUN mkdir -p llm_model && \
    curl -L "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf" \
    -o "llm_model/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define the command to run your app
# We use gunicorn for a more production-ready server
CMD ["gunicorn", "--workers", "2", "--threads", "4", "--bind", "0.0.0.0:5000", "app:app"]
