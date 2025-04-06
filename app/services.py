from sentence_transformers import SentenceTransformer

# Load the embedding model once when the app starts.
model = SentenceTransformer('all-MiniLM-L6-v2')

def compute_embedding(text: str):
    return model.encode(text).tolist()

def format_description(description: str, max_length: int = 500) -> str:
    if description and len(description) > max_length:
        return description[:max_length] + "..."
    return description
