from src.embed import cosine, embed


def test_embedding_dimension():
    vectors = embed(["hello world"])
    assert len(vectors) == 1
    assert len(vectors[0]) == 384

def test_one_vector_per_text():
    vectors = embed(["a", "b", "c"])
    assert len(vectors) == 3

def test_similar_scores_higher_than_unrelated():
    a, b, c = embed([
        "How do I reset my password?",
        "I forgot my login credentials",
        "The weather is sunny today",
    ])
    assert cosine(a, b) > cosine(a, c)