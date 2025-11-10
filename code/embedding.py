import json
import time
import os
import glob
from typing import List, Dict, Any
from datetime import datetime

class DocumentEmbedder:
    def __init__(self, model_name='all-MiniLM-L6-v2', pinecone_api_key=None, index_name="zabito-legal-index"):
        self.model_name = model_name
        self.pinecone_api_key = pinecone_api_key
        self.index_name = index_name
        self.model = None
        self.pc = None
        self.index = None
        
    def load_model(self):
        """Load the sentence transformer model"""
        try:
            from sentence_transformers import SentenceTransformer
            print(f"Loading model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print("Model loaded successfully!")
        except ImportError:
            raise ImportError("sentence-transformers not installed. Run: pip install sentence-transformers")
        except Exception as e:
            raise Exception(f"Error loading model: {e}")
    
    def initialize_pinecone(self):
        """Initialize Pinecone connection"""
        try:
            from pinecone import Pinecone, ServerlessSpec
            print("Initializing Pinecone (v3+)...")
            
            if not self.pinecone_api_key:
                raise ValueError("No Pinecone API key provided")
            
            # Initialize Pinecone client
            self.pc = Pinecone(api_key=self.pinecone_api_key)
            
            # Check if index exists, create if not
            existing_indexes = self.pc.list_indexes()
            index_names = [idx.name for idx in existing_indexes.indexes] if hasattr(existing_indexes, 'indexes') else []
            
            if self.index_name not in index_names:
                print(f"Creating new index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=384,  # For all-MiniLM-L6-v2
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",  # or "gcp" or "azure"
                        region="us-east-1"  # Change region as needed
                    )
                )
                print("Index created successfully!")
                # Wait for index to be ready
                print("Waiting for index to be ready...")
                time.sleep(30)  # Increased wait time for serverless index
            else:
                print(f"Index '{self.index_name}' already exists")
            
            # Connect to index
            self.index = self.pc.Index(self.index_name)
            print("Pinecone initialized successfully!")
            
        except ImportError:
            raise ImportError("pinecone-client not installed. Run: pip install pinecone")
        except Exception as e:
            raise Exception(f"Error initializing Pinecone: {e}")
    
    def generate_embeddings(self, chunks: List[Dict]) -> List[Dict]:
        """Generate embeddings for text chunks"""
        if not self.model:
            self.load_model()
        
        print(f"Generating embeddings for {len(chunks)} chunks...")
        
        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        for i, chunk in enumerate(chunks):
            chunk['embedding'] = embeddings[i].tolist()
            chunk['embedding_dim'] = len(embeddings[i])
            chunk['processed_at'] = datetime.now().isoformat()
        
        print("Embeddings generated successfully!")
        return chunks
    
    def store_in_pinecone(self, chunks_with_embeddings: List[Dict], batch_size=100):
        """Store embeddings in Pinecone vector database"""
        if not self.index:
            self.initialize_pinecone()
        
        print(f"Storing {len(chunks_with_embeddings)} vectors in Pinecone...")
        
        vectors = []
        for i, chunk in enumerate(chunks_with_embeddings):
            vector_id = f"chunk_{chunk.get('file_name', i)}_{int(time.time())}"
            metadata = {
                "text": chunk['text'][:1000],
                "word_count": chunk.get('word_count', 0),
                "char_count": chunk.get('char_count', 0),
                "file_name": chunk.get('file_name', 'unknown'),
                "chunk_index": i,
                "timestamp": time.time(),
                "processed_at": chunk.get('processed_at', '')
            }
            
            vectors.append({
                "id": vector_id,
                "values": chunk['embedding'],
                "metadata": metadata
            })
        
        total_vectors = len(vectors)
        successful_vectors = 0

        for i in range(0, total_vectors, batch_size):
            batch = vectors[i:i + batch_size]
            try:
                upsert_response = self.index.upsert(vectors=batch)
                successful_vectors += len(batch)
                print(f"✓ Upserted batch {i//batch_size + 1}/{(total_vectors-1)//batch_size + 1} - {len(batch)} vectors")
            except Exception as e:
                print(f"✗ Failed to upsert batch {i//batch_size + 1}: {e}")
        
        print(f"Successfully stored {successful_vectors}/{total_vectors} vectors in Pinecone!")
        
        # Print index stats
        try:
            index_stats = self.index.describe_index_stats()
            print(f"\n📊 Index Statistics:")
            print(f"   Total vectors: {index_stats['total_vector_count']}")
            print(f"   Dimension: {index_stats['dimension']}")
        except Exception as e:
            print(f"Could not retrieve index stats: {e}")

    def process_chunks(self, chunks: List[Dict], store_in_pinecone=True):
        """
        Complete pipeline: Generate embeddings and store in Pinecone
        """
        print("Starting embedding generation...")
        
        # Generate embeddings
        chunks_with_embeddings = self.generate_embeddings(chunks)
        
        # Store in Pinecone if requested and API key is available
        if store_in_pinecone and self.pinecone_api_key:
            print("\n🔄 Storing embeddings in Pinecone...")
            self.store_in_pinecone(chunks_with_embeddings)
        elif store_in_pinecone and not self.pinecone_api_key:
            print("\n⚠️  Pinecone storage skipped: No API key provided")
        else:
            print("\n⏸️  Pinecone storage disabled")
        
        return chunks_with_embeddings
    
# Simple configuration - DIRECT APPROACH
def get_pinecone_config():
    """Get Pinecone configuration directly"""
    # REPLACE THIS WITH YOUR ACTUAL PINECONE API KEY
    PINECONE_API_KEY = "pcsk_UfHqv_QJNAVD1nz7ZxAXhdMS75q2Def87Ty9xcUh7qBXi6GrET4W8WHPEcWJvTMCiyJdG"
    INDEX_NAME = "zabito-legal-index"
    
    # Validate the API key
    if not PINECONE_API_KEY or len(PINECONE_API_KEY) < 20:
        print("❌ INVALID PINECONE API KEY")
        print("Please replace the PINECONE_API_KEY with your actual key from https://app.pinecone.io/")
        return None
    
    print("✅ Valid Pinecone API key found")
    return {
        'api_key': PINECONE_API_KEY,
        'index_name': INDEX_NAME
    }

# Enhanced utility functions with append/overwrite options
def save_embeddings_locally(chunks_with_embeddings, filename="embeddings_metadata.json", mode="overwrite"):
    """
    Save embeddings metadata to local JSON file
    
    Args:
        chunks_with_embeddings: List of chunks with embeddings
        filename: Output filename
        mode: "overwrite" or "append"
    """
    # Prepare data for saving (remove embeddings to reduce file size)
    new_data = []
    for chunk in chunks_with_embeddings:
        save_chunk = {k: v for k, v in chunk.items() if k != 'embedding'}
        save_chunk['has_embedding'] = True
        new_data.append(save_chunk)
    
    if mode == "append" and os.path.exists(filename):
        # Load existing data
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            
            # Check if existing_data is a list
            if isinstance(existing_data, list):
                # Append new data to existing data
                combined_data = existing_data + new_data
                print(f"Appended {len(new_data)} chunks to existing {len(existing_data)} chunks")
            else:
                # If existing data is not a list, start fresh
                combined_data = new_data
                print("Existing file was not a list, starting fresh...")
                
        except (json.JSONDecodeError, Exception) as e:
            print(f"Error reading existing file, starting fresh: {e}")
            combined_data = new_data
    else:
        # Overwrite mode or file doesn't exist
        combined_data = new_data
        if mode == "append":
            print("File doesn't exist, creating new file...")
        else:
            print("Overwrite mode: creating new file...")
    
    # Save the data
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(combined_data)} chunks to {filename} (mode: {mode})")

def load_existing_embeddings_metadata(filename="embeddings_metadata.json"):
    """Load existing embeddings metadata to check what's already processed"""
    if not os.path.exists(filename):
        return []
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            print(f"Found {len(data)} existing chunks in {filename}")
            return data
        else:
            print("Existing file is not a list, returning empty")
            return []
    except Exception as e:
        print(f"Error loading existing embeddings: {e}")
        return []

def get_processed_files(embeddings_metadata):
    """Get list of files that have already been processed"""
    processed_files = set()
    for chunk in embeddings_metadata:
        if 'file_name' in chunk:
            processed_files.add(chunk['file_name'])
    return processed_files

def filter_unprocessed_chunks(chunks, processed_files):
    """Filter out chunks that have already been processed"""
    unprocessed_chunks = []
    for chunk in chunks:
        file_name = chunk.get('file_name', '')
        if file_name not in processed_files:
            unprocessed_chunks.append(chunk)
        else:
            print(f"Skipping already processed: {file_name}")
    
    return unprocessed_chunks

# Enhanced chunk loading functions
def load_chunks_from_txt_files(directory=".", pattern="chunk_*.txt", skip_processed=True):
    """
    Load chunks from individual text files with option to skip processed files
    """
    chunk_files = glob.glob(os.path.join(directory, pattern))
    chunk_files.sort()
    
    # Check which files are already processed if skip_processed is True
    processed_files = set()
    if skip_processed:
        existing_metadata = load_existing_embeddings_metadata()
        processed_files = get_processed_files(existing_metadata)
        print(f"Found {len(processed_files)} already processed files")
    
    chunks = []
    for file_path in chunk_files:
        file_name = os.path.basename(file_path)
        
        # Skip if already processed
        if skip_processed and file_name in processed_files:
            print(f"Skipping already processed: {file_name}")
            continue
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            
            if text:
                chunk_data = {
                    'text': text,
                    'word_count': len(text.split()),
                    'char_count': len(text),
                    'file_name': file_name,
                    'file_path': file_path
                }
                chunks.append(chunk_data)
                
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    print(f"Loaded {len(chunks)} unprocessed chunks from {len(chunk_files)} total files")
    return chunks

def test_pinecone_connection():
    """Test Pinecone connection"""
    config = get_pinecone_config()
    if not config:
        return False
    
    try:
        from pinecone import Pinecone
        print("🔗 Testing Pinecone connection (v3+)...")
        
        # Initialize Pinecone client
        pc = Pinecone(api_key=config['api_key'])
        
        # List indexes to test connection
        indexes = pc.list_indexes()
        index_names = [idx.name for idx in indexes.indexes] if hasattr(indexes, 'indexes') else []
        print(f"✅ Pinecone connection successful!")
        print(f"Available indexes: {index_names}")
        
        # Check if our index exists
        if config['index_name'] in index_names:
            print(f"✅ Index '{config['index_name']}' exists")
            index = pc.Index(config['index_name'])
            stats = index.describe_index_stats()
            print(f"Index stats: {stats}")  
        else:
            print(f"⚠️ Index '{config['index_name']}' does not exist yet")
        
        return True
        
    except Exception as e:
        print(f"❌ Pinecone connection failed: {e}")
        return False

def load_chunks_from_single_txt(file_path, skip_if_processed=True):
    """
    Load a single text file as one chunk with processing check
    """
    file_name = os.path.basename(file_path)
    
    # Check if already processed
    if skip_if_processed:
        existing_metadata = load_existing_embeddings_metadata()
        processed_files = get_processed_files(existing_metadata)
        if file_name in processed_files:
            print(f"File {file_name} already processed, skipping...")
            return []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        
        if text:
            chunks = [{
                'text': text,
                'word_count': len(text.split()),
                'char_count': len(text),
                'file_name': file_name,
                'file_path': file_path
            }]
            print(f"Loaded 1 chunk from {file_path}")
            return chunks
        else:
            print(f"File {file_path} is empty")
            return []
            
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []

# Main execution function with flexible options
def main(file_pattern="chunk_001.txt", mode="append", skip_processed=True, use_pinecone=True):
    """
    Main function to run the embedding pipeline
    
    Args:
        file_pattern: File or pattern to process ("chunk_001.txt" or "chunk_*.txt")
        mode: "append" or "overwrite" for saving embeddings
        skip_processed: Whether to skip already processed files
        use_pinecone: Whether to store in Pinecone
    """
    
    # Configuration
    PINECONE_API_KEY = "pcsk_UfHqv_QJNAVD1nz7ZxAXhdMS75q2Def87Ty9xcUh7qBXi6GrET4W8WHPEcWJvTMCiyJdG" if use_pinecone else None
    
    print("=== DOCUMENT EMBEDDING PIPELINE ===")
    print(f"Mode: {mode}, Skip Processed: {skip_processed}, Use Pinecone: {use_pinecone}")
    
    print("1. Testing Pinecone connection...")
    if not test_pinecone_connection():
        print("Cannot continue without Pinecone connection")
        return
    
    config = get_pinecone_config()
    if not config:
        return

    print("\n2. Loading chunks...")

    # Load chunks based on pattern
    if "*" in file_pattern:
        # Pattern matching (multiple files)
        chunks = load_chunks_from_txt_files(pattern=file_pattern, skip_processed=skip_processed)
    else:
        # Single file
        chunks = load_chunks_from_single_txt(file_pattern, skip_if_processed=skip_processed)
    
    if not chunks:
        print("No chunks found to process.")
        return []
    
    print(f"Processing {len(chunks)} chunks...")
    
    # Initialize embedder
    embedder = DocumentEmbedder(
        model_name='all-MiniLM-L6-v2',
        pinecone_api_key=config['api_key'],
        index_name=config['index_name']
    )
    
    # Generate embeddings
    print("\n3. Generating embeddings...")
    chunks_with_embeddings = embedder.process_chunks(chunks, store_in_pinecone=True)
    
    # # Store in Pinecone if enabled
    # if use_pinecone and PINECONE_API_KEY and PINECONE_API_KEY != "pcsk_UfHqv_QJNAVD1nz7ZxAXhdMS75q2Def87Ty9xcUh7qBXi6GrET4W8WHPEcWJvTMCiyJdG":
    #     print("\n3. Storing in Pinecone...")
    #     embedder.store_in_pinecone(chunks_with_embeddings)
    # else:
    #     print("\n3. Skipping Pinecone storage")
    
    # Save embeddings locally
    print("\n4. Saving embeddings metadata locally...")
    save_embeddings_locally(chunks_with_embeddings, mode=mode)
    
    print("\n=== EMBEDDING GENERATION COMPLETE ===")
    print(f"Processed {len(chunks_with_embeddings)} chunks")
    
    if chunks_with_embeddings:
        print(f"   Embedding dimension: {chunks_with_embeddings[0]['embedding_dim']}")

    return chunks_with_embeddings

if __name__ == "__main__":
    # Different usage examples:
    
    # Example 1: Process chunk_002.txt and append to existing embeddings
    # main(file_pattern="chunk_002.txt", mode="append", skip_processed=True, use_pinecone=False)
    
    # Example 2: Process all chunks and overwrite existing file
    # main(file_pattern="chunk_*.txt", mode="overwrite", skip_processed=False, use_pinecone=False)
    
    # Example 3: Process only unprocessed chunks and append
    main(file_pattern="chunk_*.txt", mode="append", skip_processed=True, use_pinecone=True)
    
    # Default: Process chunk_001.txt in append mode
    # main(file_pattern="chunk_005.txt", mode="append", skip_processed=True, use_pinecone=False)