import warnings
warnings.filterwarnings('ignore')

from config import config
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import pickle

class EmbeddingsWrapper:
    """Wrapper for sentence-transformers embeddings"""
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def embed_query(self, text):
        return self.model.encode(text).tolist()
    
    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

def load_chunks():
    """Load chunks from disk"""
    chunks_path = config.DATA_PATH / "chunks.pkl"
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks not found at: {chunks_path}")
    
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
    print(f"Loaded {len(chunks)} chunks from disk")
    return chunks

def create_vector_store(chunks, recreate=True):
    """Create ChromaDB vector store"""
    print("Creating vector store...")
    
    embeddings = EmbeddingsWrapper(config.EMBEDDING_MODEL)
    
    client = chromadb.PersistentClient(
        path=str(config.VECTOR_DB_PATH),
        settings=Settings(anonymized_telemetry=False)
    )
    
    if recreate:
        try:
            client.delete_collection(config.COLLECTION_NAME)
            print("   Cleared existing collection")
        except:
            pass
    
    collection = client.create_collection(
        name=config.COLLECTION_NAME,
        metadata={"model": config.EMBEDDING_MODEL}
    )
    
    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        
        ids = [chunk.metadata['chunk_id'] for chunk in batch]
        documents = [chunk.page_content for chunk in batch]
        metadatas = [chunk.metadata for chunk in batch]
        
        batch_embeddings = embeddings.model.encode(documents).tolist()
        
        collection.add(
            ids=ids,
            embeddings=batch_embeddings,
            documents=documents,
            metadatas=metadatas
        )
        print(f"   Added batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}")
    
    print(f"Vector store created with {collection.count()} documents")
    print(f" Location: {config.VECTOR_DB_PATH}")
    
    return collection, embeddings

def test_vector_store(collection, embeddings):
    
    print("\n Testing vector store...")
    
    test_queries = [
        "Amazon revenue 2022",
        "risk factors",
        "number of employees",
        "AWS growth"
    ]
    
    for query in test_queries:
        print(f"\n   Query: '{query}'")
        query_vec = embeddings.embed_query(query)
        results = collection.query(
            query_embeddings=[query_vec],
            n_results=2,
            include=["documents", "metadatas", "distances"]
        )
        
        print(f"   Found {len(results['documents'][0])} results")
        for i, (doc, meta, dist) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        )):
            print(f"     {i+1}. Page {meta.get('page')} (dist: {dist:.3f})")

def main():
    """Main function to create vector store"""
    print("=" * 50)
    print("Amazon 10-K Vector Store Creator")
    print("=" * 50)
    
    try:
        chunks = load_chunks()
        
        collection, embeddings = create_vector_store(chunks)
        
        test_vector_store(collection, embeddings)
        
        return collection, embeddings
        
    except Exception as e:
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    main()