from flask import Flask, render_template, request, jsonify, send_file
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import faiss
import io
import os
from database import init_db, insert_chat, chat_history
from read_documents import read_and_split_pdfs

app = Flask(__name__)
app.secret_key = "supersecretkey"
init_db()

# Step 1: Load Models for Offline Mode
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.1', local_files_only=True)
model = AutoModelForCausalLM.from_pretrained('mistralai/Mistral-7B-Instruct-v0.1', device_map='auto', torch_dtype=torch.float16, local_files_only=True)

# Step 2: Initialize FAISS
d = 384  # Dimension of embeddings (MiniLM-L6-v2)
documents_dir = os.path.join(os.path.dirname(__file__), "documents")
vector_store_path = os.path.join(os.path.dirname(__file__), "vector_store")
vector_store, _, _, _, _, _ = read_and_split_pdfs(documents_dir, vector_store_path)

# Step 3: Retrieve Documents
def retrieve_documents(query, top_k=5):
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    distances, indices = vector_store.index.search(query_embedding, top_k)
    return [vector_store.texts[i] for i in indices[0]]

# Step 4: Generate Response using LLM
def generate_response(query):
    context = '\n'.join(retrieve_documents(query))
    input_text = f"Context: {context}\nQuestion: {query}\nAnswer:"
    inputs = tokenizer(input_text, return_tensors='pt').to('cuda')
    outputs = model.generate(**inputs, max_new_tokens=150)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message', '')
    if not user_input:
        return jsonify({'response': 'No input provided.'})
    response = generate_response(user_input)
    insert_chat(user_input, response)
    return jsonify({'response': response})

@app.route('/documents/<filename>')
def serve_pdf(filename):
    file_path = os.path.join(documents_dir, filename)
    if os.path.exists(file_path):
        with open(file_path, 'rb') as bites:
            return send_file(io.BytesIO(bites.read()), download_name=filename, mimetype='application/pdf')
    return jsonify({'error': 'File not found'}), 404

if __name__ == '__main__':
    app.run(debug=False)





1. Add Required Packages to requirements.txt


To run the application offline, ensure you have the following packages in your `requirements.txt` file:
```plaintext
flask
torch
transformers
sentence-transformers
faiss-cpu
numpy
If you have a GPU, replace faiss-cpu with faiss-gpu.

Ensure torch is added with the correct version for your hardware (CUDA or CPU).

2. Pre-Download Models for Offline Use
You can pre-download the models and store them locally:

For SentenceTransformer (MiniLM-L6-v2):

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
model.save('models/all-MiniLM-L6-v2')
For Mistral-7B-Instruct:

from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.1')
tokenizer.save_pretrained('models/Mistral-7B-Instruct-v0.1')
model = AutoModelForCausalLM.from_pretrained('mistralai/Mistral-7B-Instruct-v0.1')
model.save_pretrained('models/Mistral-7B-Instruct-v0.1')
3. Update Your Code to Load Models from Local Path
Modify your code to load the models from the downloaded directory:


embedding_model = SentenceTransformer('models/all-MiniLM-L6-v2')
tokenizer = AutoTokenizer.from_pretrained('models/Mistral-7B-Instruct-v0.1', local_files_only=True)
model = AutoModelForCausalLM.from_pretrained('models/Mistral-7B-Instruct-v0.1', device_map='auto', torch_dtype=torch.float16, local_files_only=True)
