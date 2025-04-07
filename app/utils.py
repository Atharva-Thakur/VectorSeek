from sentence_transformers import SentenceTransformer


def embed_text(text: str):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model.encode([text])[0].tolist()