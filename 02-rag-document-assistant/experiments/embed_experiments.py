from src.embed import cosine, embed

pairs = [
    ("Same meaning, different words",
     "How do I reset my password?", "I forgot my login credentials"),
    ("Same words, different meaning",
     "I deposited money at the bank", "We sat on the river bank"),
    ("Unrelated",
     "The weather is nice", "Python is a programming language"),
]
for label, a, b in pairs:
    va, vb = embed([a, b])
    print(f"{label}: {cosine(va, vb):.3f}")