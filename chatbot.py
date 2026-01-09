import warnings
warnings.filterwarnings('ignore')

from config import config
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

class EmbeddingsWrapper:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def embed_query(self, text):
        return self.model.encode(text).tolist()

class AmazonChatBot:
    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=str(config.VECTOR_DB_PATH),
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_collection(config.COLLECTION_NAME)
        
        self.embeddings = EmbeddingsWrapper(config.EMBEDDING_MODEL)
        
        self.chat_history = []
        
        print(f"ChatBot initialized with {self.collection.count()} documents")
    
    def search(self, query, k=5):
        query_vec = self.embeddings.embed_query(query)
        
        results = self.collection.query(
            query_embeddings=[query_vec],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )
        
        return results
    
    def answer(self, question, k=3):
        results = self.search(question, k)
        
        if results['documents'][0]:
            best_match = results['documents'][0][0]
            best_metadata = results['metadatas'][0][0]
            best_distance = results['distances'][0][0]
            
            page = best_metadata.get('page', 'unknown')
            item = best_metadata.get('item', 'unknown')
            
            answer = f"""Based on Page {page} (Item {item}) of Amazon's 10-K report:

{best_match[:600]}...

[Similarity: {1-best_distance:.2%}]"""
            
            source_pages = set()
            for metadata in results['metadatas'][0]:
                source_pages.add(str(metadata.get('page', '?')))
        else:
            answer = "No relevant information found in the document."
            source_pages = set()
        
        self.chat_history.append((question, answer))
        
        return {
            'answer': answer,
            'sources': list(source_pages),
            'chunks_used': len(results['documents'][0])
        }
    
    def clear_history(self):
        self.chat_history = []
        return "History cleared"

def main():
    print("=" * 50)
    print("Amazon 10-K ChatBot Test")
    print("=" * 50)
    
    try:
        chatbot = AmazonChatBot()
        
        test_questions = [
            "What was Amazon's revenue in 2022?",
            "How many employees does Amazon have?"
        ]
        
        for question in test_questions:
            print(f"\n Question: {question}")
            result = chatbot.answer(question)
            print(f"💡 Answer: {result['answer']}")
            if result['sources']:
                print(f" Sources: Pages {', '.join(result['sources'])}")
        
        print("\n ChatBot test successful!")
        return chatbot
        
    except Exception as e:
        print(f" Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()