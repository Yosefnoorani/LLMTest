from flask import Flask, request, jsonify
import subprocess
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
app.debug = True

# ========================
# 1. Run Embedding for Files
# ========================
UPLOAD_FOLDER = "uploaded_files"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/generate_embeddings', methods=['POST'])
def run_embedding():
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No files part in the request'}), 400

        files = request.files.getlist('files')
        file_paths = []

        for file in files:
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)
            file_paths.append(file_path)

        if not file_paths:
            return jsonify({'error': 'No files uploaded'}), 400

        # עטיפת נתיבי הקבצים במרכאות
        embedding_script = 'multi_file_embedding.py'
        command = ['python', embedding_script] + [f'"{file}"' for file in file_paths]

        # שימוש ב-subprocess להרצת הסקריפט
        result = subprocess.run(command, text=True, capture_output=True)

        if result.returncode != 0:
            error_message = result.stderr or "Unknown error occurred."
            return jsonify({'error': f"Embedding script failed: {error_message}"}), 500

        return jsonify({'message': 'Embeddings created successfully.'}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ========================
# 2. Query Embedding
# ========================
@app.route('/query', methods=['POST'])
def query_embedding():
    try:
        data = request.get_json()
        question = data.get('question', '')

        if not question:
            return jsonify({'error': 'No question provided'}), 400

        # Call the query script with the question as an argument
        query_script = 'chat_query_embedding.py'
        command = ['python', query_script, '--query', question]
        result = subprocess.run(command, capture_output=True, text=True, check=True)

        # Parse the output and send the response back
        return jsonify({'response': result.stdout.strip()}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ========================
# 3. Run the Flask App
# ========================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
