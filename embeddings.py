from langchain_community.embeddings import OllamaEmbeddings


def generate_embeddings():
    # embedder = OllamaEmbeddings(model="llama3")
    embedder = OllamaEmbeddings(model="mxbai-embed-large")
    return embedder