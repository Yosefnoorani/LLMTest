from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename
from flask_cors import CORS
from multi_file_embedding import embedding_files_multiple_dirs
from chat_query_embedding import query_embedding

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route('/upload', methods=['POST'])
def upload_files():
    try:
        files = request.files.getlist('files')
        if not files:
            return jsonify({"error": "No files provided."}), 400

        file_paths = []
        for file in files:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            file_paths.append(file_path)

        return jsonify({"message": "Files uploaded successfully.", "file_paths": file_paths}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/generate_embeddings', methods=['POST'])
def generate_embeddings():
    try:
        data = request.get_json()
        file_paths = data.get('file_paths')
        if not file_paths:
            return jsonify({"error": "File paths missing."}), 400

        # Extract the directories from the file paths
        directories = list(set([os.path.dirname(file_path) for file_path in file_paths]))

        output_file = os.path.join(app.config['UPLOAD_FOLDER'], 'embeddings_with_overlap_llama.json')

        # Call the function from multi_file_embedding.py
        embedding_files_multiple_dirs(directories, output_file)

        return jsonify({"message": "Embeddings generated successfully.", "output_file": output_file}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/query', methods=['POST'])
def query():
    try:
        data = request.get_json()
        json_file = data.get('json_file')
        query_text = data.get('query')

        if not json_file or not query_text:
            return jsonify({"error": "Missing json_file or query."}), 400

        # Call the function from chat_query_embedding.py
        answer = query_embedding(query_text, json_file)

        return jsonify({"answer": answer}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
