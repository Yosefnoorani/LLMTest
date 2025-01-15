from flask import Flask, request, jsonify
import os
import json
from werkzeug.utils import secure_filename
from multi_file_embedding import embedding_files_multiple_dirs
from chat_query_embedding import query_embedding
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# @app.route('/generate_embeddings', methods=['POST'])
# def generate_embeddings():
#     try:
#         # Save uploaded files to the server
#         files = request.files.getlist('files')
#         if not files:
#             return jsonify({"error": "No files provided."}), 400
#
#         file_paths = []
#         for file in files:
#             filename = secure_filename(file.filename)
#             file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#             file.save(file_path)
#             file_paths.append(file_path)
#
#         # Generate embeddings
#         output_file = os.path.join(app.config['UPLOAD_FOLDER'], 'embeddings_with_overlap_llama.json')
#         embedding_files_multiple_dirs([app.config['UPLOAD_FOLDER']], output_file=output_file)
#
#         return jsonify({"message": "Embeddings generated successfully.", "output_file": output_file}), 200
#
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

@app.route('/generate_embeddings', methods=['POST'])
def generate_embeddings():
    try:
        # קבלת הקבצים שהועלו
        files = request.files.getlist('files')
        if not files:
            return jsonify({"error": "No files provided."}), 400

        # יצירת רשימה לשמירת הנתיבים
        file_paths = []
        for file in files:
            # שם קובץ מאובטח
            filename = secure_filename(file.filename)
            # נתיב יעד
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            # שמירת הקובץ בנתיב
            file.save(file_path)
            file_paths.append(file_path)

        print(f"Uploaded files saved at: {file_paths}")

        # קריאה לפונקציית יצירת ה-embeddings
        # במקום לעבור רק על תיקייה אחת, נוודא שעוברים על כל הקבצים שהועלו
        output_file = os.path.join(app.config['UPLOAD_FOLDER'], 'embeddings_with_overlap_llama.json')
        embedding_files_multiple_dirs([os.path.dirname(file_path) for file_path in file_paths], output_file=output_file)

        return jsonify({"message": "Embeddings generated successfully.", "output_file": output_file}), 200

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/query', methods=['POST'])
def query():
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({"error": "Query parameter missing."}), 400

        query_text = data['query']
        response = query_embedding(query=query_text)

        return jsonify({"answer": response}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
