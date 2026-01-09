import warnings
warnings.filterwarnings('ignore')

from config import config
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import hashlib
import re
import pickle

def load_pdf():
    """Load and parse PDF document"""
    print(f"📄 Loading PDF: {config.DOCUMENT_PATH}")
    
    if not config.DOCUMENT_PATH.exists():
        raise FileNotFoundError(f"PDF not found at: {config.DOCUMENT_PATH}")
    
    loader = PyPDFLoader(str(config.DOCUMENT_PATH))
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} pages")
    return documents

def create_chunks(documents):
    """Split documents into chunks with metadata"""
    print("✂️  Creating chunks...")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    all_chunks = []
    for page_idx, doc in enumerate(documents):
        page_num = page_idx + 1
        text = doc.page_content
        
        item_match = re.search(r'Item\s+(\d+[A-Z]*)', text[:500], re.IGNORECASE)
        item_num = item_match.group(1) if item_match else "unknown"
        
        page_chunks = splitter.split_text(text)
        
        for chunk_idx, chunk_text in enumerate(page_chunks):
            metadata = {
                'page': page_num,
                'chunk_id': f"p{page_num}_c{chunk_idx}",
                'chunk_index': chunk_idx,
                'source': str(config.DOCUMENT_PATH.name),
                'item': item_num,
                'text_hash': hashlib.md5(chunk_text.encode()).hexdigest()[:8]
            }
            
            all_chunks.append(Document(
                page_content=chunk_text,
                metadata=metadata
            ))
    
    print(f"Created {len(all_chunks)} chunks")
    
    avg_size = sum(len(c.page_content) for c in all_chunks) / len(all_chunks)
    print(f"   Average chunk size: {avg_size:.0f} characters")
    
    return all_chunks

def save_chunks(chunks):
    """Save chunks to disk"""
    chunks_path = config.DATA_PATH / "chunks.pkl"
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)
    print(f"💾 Saved chunks to: {chunks_path}")
    return chunks_path

def main():
    """Main function to load PDF and create chunks"""
    print("=" * 50)
    print("Amazon 10-K Data Loader")
    print("=" * 50)
    
    try:
        documents = load_pdf()
        
        chunks = create_chunks(documents)
        
        save_chunks(chunks)
        
        print("\n🎉 Data loading complete!")
        return chunks
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise

if __name__ == "__main__":
    main()