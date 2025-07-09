from langchain.embeddings import OllamaEmbeddings
ollama_endpoint = "http://127.0.0.1:11434"

def test_ollama_connection():
    try:
        # Initialize the embedding model
        embeddings = OllamaEmbeddings(model="nomic-embed-text")

        # Test embedding on a sample text
        result = embeddings.embed_query("Hello from Ollama!")

        print("✅ Ollama is working! Embedding result:")
        print(result[:5])  # Print first few values for brevity
    except Exception as e:
        print("❌ Failed to connect to Ollama.")
        print("Error:", e)

if __name__ == "__main__":
    test_ollama_connection()
