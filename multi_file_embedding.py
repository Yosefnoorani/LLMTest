import os
import json
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import SimpleDirectoryReader

# ========================
# 1. Set up OpenAI API
# ========================
os.environ["OPENAI_API_KEY"] = 'YOUR_KEY'

# ========================
# 2. Load Documents
# ========================
def load_documents(directory_path):
    """Load documents from a folder"""
    loader = SimpleDirectoryReader(directory_path)
    return loader.load_data()

# ========================
# 3. Chunk Documents with Overlap
# ========================
def chunk_document_with_overlap(document, chunk_size=200, overlap=10):
    """
    Split a document into smaller chunks with overlap.

    Parameters:
    - document (str): The text of the document to chunk.
    - chunk_size (int): The maximum size of each chunk in words.
    - overlap (int): The number of overlapping words between consecutive chunks.

    Returns:
    - List[str]: A list of text chunks.
    """
    words = document.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
        if i + chunk_size >= len(words):
            break

    return chunks

# ========================
# 4. Generate Embeddings with OpenAIEmbedding
# ========================
def generate_embeddings_with_llama(text_chunks):
    """Generate embeddings for a list of text chunks using OpenAIEmbedding"""
    openai_embed = OpenAIEmbedding()
    embeddings = []
    for chunk in text_chunks:
        embedding = openai_embed.get_text_embedding(chunk)
        embeddings.append({
            "chunk": chunk,
            "embedding": embedding
        })
    return embeddings

# ========================
# 5. Save Embeddings to JSON
# ========================
def save_embeddings_to_json(embeddings, file_path):
    """Save embeddings to a JSON file"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(embeddings, f, ensure_ascii=False, indent=4)

# ========================
# Main Process
# ========================
# def embedding_files_multiple_dirs(directory_paths, output_file="embeddings_with_overlap_llama.json"):
#     """
#     Process multiple directories of files and save their embeddings.
#
#     Parameters:
#     - directory_paths (list): List of directories containing files.
#     - output_file (str): Path to save the embeddings JSON file.
#     """
#     all_embeddings = []
#
#     for directory_path in directory_paths:
#         if not os.path.exists(directory_path):
#             print(f"Directory {directory_path} does not exist. Skipping.")
#             continue
#
#         # Load documents
#         documents = load_documents(directory_path)
#
#         # Process each document
#         for document in documents:
#             # Split document into chunks with overlap
#             chunks = chunk_document_with_overlap(document.text, chunk_size=200, overlap=10)
#             # Generate embeddings for each chunk
#             embeddings = generate_embeddings_with_llama(chunks)
#
#             # Add document name to each embedding
#             for embedding in embeddings:
#                 embedding["document_name"] = document.doc_id
#
#             all_embeddings.extend(embeddings)
#
#     # Save all embeddings to a JSON file
#     save_embeddings_to_json(all_embeddings, output_file)
#
#     print(f"Embeddings saved to {output_file}.")

def embedding_files_multiple_dirs(directory_paths, output_file="embeddings_with_overlap_llama.json"):
    """
    Process multiple directories of files and save their embeddings.

    Parameters:
    - directory_paths (list): List of directories containing files.
    - output_file (str): Path to save the embeddings JSON file.
    """
    all_embeddings = []

    for directory_path in directory_paths:
        if not os.path.exists(directory_path):
            print(f"Directory {directory_path} does not exist. Skipping.")
            continue

        # Load documents
        documents = load_documents(directory_path)

        # Process each document
        for document in documents:
            # Split document into chunks with overlap
            chunks = chunk_document_with_overlap(document.text, chunk_size=200, overlap=10)
            # Generate embeddings for each chunk
            embeddings = generate_embeddings_with_llama(chunks)

            # Add document name to each embedding
            for embedding in embeddings:
                embedding["document_name"] = document.doc_id

            all_embeddings.extend(embeddings)

    # Save all embeddings to a JSON file
    save_embeddings_to_json(all_embeddings, output_file)

    print(f"Embeddings saved to {output_file}.")

# Example usage:
# embedding_files_multiple_dirs([r"C:\\Users\\yosef\\Downloads\\Portal", r"C:\Users\yosef\Downloads\Inention"])
# "C:\Users\yosef\Downloads\INTENTION BEYOND - V2.pdf"

import sys


if __name__ == "__main__":
    try:
        file_paths = sys.argv[1:]  # Get file paths passed as arguments
        print(f"Received file paths: {file_paths}")

        if not file_paths:
            raise ValueError("No file paths provided.")

        # Continue with your embedding logic...
        print("Embedding process started...")
        # Placeholder for actual embedding process
        for file in file_paths:
            print(f"Processing file: {file}")
        print("Embedding process completed.")

    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)