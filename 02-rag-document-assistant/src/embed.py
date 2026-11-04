from functools import lru_cache

from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"  # 384 dimensions, free, runs locally

@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    # Loaded once, then reused. Loading is slow, encoding is fast.
    return SentenceTransformer(MODEL_NAME)

def embed(texts: list[str]) -> list[list[float]]:
    vectors = get_model().encode(texts, normalize_embeddings=True)
    return vectors.tolist()

def cosine(a: list[float], b: list[float]) -> float:
    # Vectors are normalized, so the dot product equals cosine similarity.
    return sum(x * y for x, y in zip(a, b))

if __name__ == "__main__":
    words = ["cat", "kitten", "car"]
    cat, kitten, car = embed(words)
    print("vector length:", len(cat))
    print("cat vs kitten:", round(cosine(cat, kitten), 3))
    print("cat vs car:   ", round(cosine(cat, car), 3))