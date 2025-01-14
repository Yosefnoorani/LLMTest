import os
import json
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.data_structs import Node
from llama_index.llms.openai import OpenAI
# from llama_index.embeddings.openai import OpenAIEmbedding

# ========================
# 1. Set up OpenAI API
# ========================
os.environ["OPENAI_API_KEY"] = "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"


# ========================
# 2. Define the OpenAI-based LLM
# ========================
def load_openai_llm():
    """Load the OpenAI GPT model"""
    return OpenAI(model="gpt-4o-mini")  # Change to "gpt-3.5-turbo" if needed.





# ========================
# 3. Load Embeddings from JSON
# ========================
def load_embeddings_from_json(json_file_path):
    """Load embeddings from a JSON file"""
    if not os.path.exists(json_file_path):
        raise FileNotFoundError(f"JSON file not found: {json_file_path}")

    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Convert JSON data to Document nodes
    nodes = [
        Node(
            doc_id=str(idx),
            text=entry["chunk"],
            embedding=entry["embedding"]
        )
        for idx, entry in enumerate(data)
    ]
    return nodes





# ========================
# 4. Build Index from Nodes
# ========================
def build_index_from_embeddings(nodes, llm):
    """Build an index using precomputed embeddings"""
    return VectorStoreIndex(nodes, llm=llm)





# ========================
# 5. Query the Index
# ========================
def query_index(index, query):
    """Run a query on the index"""
    response = index.as_query_engine(similarity_top_k=5).query(query)
    return response.response





# ========================
# 6. Save and Load Index
# ========================
def save_index(index, file_path):
    """Save the index to a file"""
    index.storage_context.persist(persist_dir=file_path)


# def load_saved_index(file_path):
#     """Load a saved index from a file"""
#     storage_context = StorageContext.from_defaults(persist_dir=file_path)
#     return VectorStoreIndex.from_storage(storage_context)

def query_embedding(query = "What are the key points in these documents?"):


    llm = load_openai_llm()

    # Path to the JSON file with embeddings
    json_file_path = "embeddings_with_overlap_llama.json"

    # Load embeddings
    embedding_nodes = load_embeddings_from_json(json_file_path)

    index = build_index_from_embeddings(embedding_nodes, llm)

    # Example query

    response = query_index(index, query)
    print(f"Query: {query}\nResponse: {response}")

    # Path to save the index
    index_file_path = "__custom_index"
    save_index(index, index_file_path)

    print(f"Index saved to {index_file_path}. Ready for use!")
    return response
