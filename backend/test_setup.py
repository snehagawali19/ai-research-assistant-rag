from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

text = "Hello AI world"

embedding = model.encode(text)

print("Embedding created successfully")
print("Embedding length:", len(embedding))