import json
import time
import os
import glob
from typing import List, Dict, Any
from datetime import datetime

class GeminiEmbedder:
    def __init__(self, google_api_key=None, model="gemini-embedding-001", pinecone_api_key=None, index_name="zabito-legal-gemini"):
        self.google_api_key = google_api_key
        self.model = model
        self.pinecone_api_key = pinecone_api_key
        self.index_name = index_name
        self.client = None
        self.pc = None
        self.index = None
        
    def initialize_gemini(self):
        """Initialize Gemini client"""
        try:
            import google.generativeai as genai
            print("Initializing Gemini client...")

            if not self.google_api_key:
                raise ValueError("No Google API key provided")
            genai.configure(api_key=self.google_api_key)
            self.client = genai
            print("Gemini client initialized successfully!")

        except ImportError:
            raise ImportError("google.generativeai not installed. Run: pip install google-generativeai")
        except Exception as e:
            raise Exception(f"Error initializing Gemini: {e}")

    def initialize_pinecone(self, dimension=768):
        """Initialize Pinecone connection for Gemini embeddings"""
        try:
            from pinecone import Pinecone, ServerlessSpec
            print("Initializing Pinecone for Gemini embeddings...")
            
            if not self.pinecone_api_key:
                raise ValueError("No Pinecone API key provided")
            
            # Initialize Pinecone client
            self.pc = Pinecone(api_key=self.pinecone_api_key)
            
            # Check if index exists, create if not
            existing_indexes = self.pc.list_indexes()
            index_names = [idx.name for idx in existing_indexes.indexes] if hasattr(existing_indexes, 'indexes') else []
            
            # Gemini embedding dimensions
            # embedding_dims = {
            #     "gemini-embedding-001": 3072
            # }
            
            # dimension = embedding_dims.get(self.model, 3072)  # Default to 3072 if unknown
            
            if self.index_name not in index_names:
                print(f"Creating new index for Gemini: {self.index_name}")
                print(f"Dimension: {dimension}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                print("Index created successfully!")
                print(f"Waiting for index to be ready (dimension: {dimension})...")
                time.sleep(30)
            else:
                print(f"Index '{self.index_name}' already exists")
            
            # Connect to index
            self.index = self.pc.Index(self.index_name)
            print("Pinecone initialized successfully!")
            
        except ImportError:
            raise ImportError("pinecone not installed. Run: pip install pinecone")
        except Exception as e:
            raise Exception(f"Error initializing Pinecone: {e}")
    
    def get_embedding_dimension(self):
        """Get the actual embedding dimension by testing with a small query"""
        if not self.client:
            self.initialize_gemini()
        
        try:
            # Test with a small query to get actual dimension
            import google.generativeai as genai
            test_response = genai.embed_content(
                model=self.model,
                content="test"
            )
            actual_dimension = len(test_response['embedding'])
            print(f"Detected embedding dimension: {actual_dimension}")
            return actual_dimension
        except Exception as e:
            print(f"Error detecting dimension, using default: {e}")
            return 768  # Fallback dimension

    def generate_embeddings(self, chunks: List[Dict]) -> List[Dict]:
        """Generate embeddings using Gemini API"""
        if not self.client:
            self.initialize_gemini()

        print(f"Generating Gemini embeddings for {len(chunks)} chunks...")
        print(f"Using model: {self.model}")
        
        chunks_with_embeddings = []
        
        for i, chunk in enumerate(chunks):
            try:
                # Call Gemini API
                response = self.client.embed_content(
                    model=self.model,
                    content=chunk['text']
                )
                
                embedding = response['embedding']
                
                # Add embedding to chunk
                chunk_with_embedding = chunk.copy()
                chunk_with_embedding['embedding'] = embedding
                chunk_with_embedding['embedding_dim'] = len(embedding)
                chunk_with_embedding['embedding_model'] = self.model
                chunk_with_embedding['processed_at'] = datetime.now().isoformat()
                
                chunks_with_embeddings.append(chunk_with_embedding)
                
                print(f"✓ Processed chunk {i+1}/{len(chunks)}")
                
                # Rate limiting - be nice to the API
                time.sleep(0.1)
                
            except Exception as e:
                print(f"✗ Error processing chunk {i+1}: {e}")
                # Continue with next chunk even if one fails
                continue

        print(f"Gemini embeddings generated successfully! ({len(chunks_with_embeddings)}/{len(chunks)} chunks)")
        return chunks_with_embeddings
    
    def store_in_pinecone(self, chunks_with_embeddings: List[Dict], batch_size=50):
        """Store embeddings in Pinecone vector database"""
        # First, detect the actual embedding dimension
        if chunks_with_embeddings:
            actual_dimension = chunks_with_embeddings[0]['embedding_dim']
            print(f"Actual embedding dimension from data: {actual_dimension}")
        else:
            actual_dimension = self.get_embedding_dimension()
        
        # Initialize Pinecone with correct dimension
        self.initialize_pinecone(dimension=actual_dimension)

        print(f"Storing {len(chunks_with_embeddings)} Gemini vectors in Pinecone...")

        vectors = []
        for i, chunk in enumerate(chunks_with_embeddings):
            vector_id = f"gemini_{i}_{int(time.time())}"
            metadata = {
                "text": chunk['text'][:1000],
                "word_count": chunk.get('word_count', 0),
                "char_count": chunk.get('char_count', 0),
                "file_name": chunk.get('file_name', 'unknown'),
                "embedding_model": chunk.get('embedding_model', 'unknown'),
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

        # Smaller batch size for API stability
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
        Complete pipeline: Generate embeddings with Gemini and store in Pinecone
        """
        print("Starting Gemini embedding generation...")
        
        # Generate embeddings
        chunks_with_embeddings = self.generate_embeddings(chunks)
        
        # Store in Pinecone if requested
        if store_in_pinecone and self.pinecone_api_key:
            print("\n🔄 Storing Gemini embeddings in Pinecone...")
            self.store_in_pinecone(chunks_with_embeddings)
        elif store_in_pinecone and not self.pinecone_api_key:
            print("\n⚠️  Pinecone storage skipped: No API key provided")
        else:
            print("\n⏸️  Pinecone storage disabled")
        
        return chunks_with_embeddings

# Configuration
def get_gemini_config():
    """Get Gemini configuration"""
    GOOGLE_API_KEY = "your-google-api-key"
    PINECONE_API_KEY = "your-pinecone-api-key"
    MODEL = "gemini-embedding-001"  # Options: "text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"
    INDEX_NAME = "your-index-name"
    
    # Validate API keys
    if not GOOGLE_API_KEY or len(GOOGLE_API_KEY) < 20:
        print("❌ INVALID GEMINI API KEY")
        return None
    
    if not PINECONE_API_KEY or len(PINECONE_API_KEY) < 20:
        print("❌ INVALID PINECONE API KEY")
        return None

    print("✅ Valid Gemini API key found")
    print(f"📊 Using model: {MODEL}")
    
    return {
        'gemini_api_key': GOOGLE_API_KEY,
        'pinecone_api_key': PINECONE_API_KEY,
        'model': MODEL,
        'index_name': INDEX_NAME
    }

def test_gemini_connection():
    """Test Gemini connection"""
    config = get_gemini_config()
    if not config:
        return False
    
    try:
        import google.generativeai as genai
        print("🔗 Testing Gemini connection...")

        genai.configure(api_key=config['gemini_api_key'])
        
        # Test with a small request
        response = genai.embed_content(
            model=config['model'],
            content="Test connection"
        )

        print(f"✅ Gemini connection successful!")
        print(f"   Model: {config['model']}")
        print(f"   Embedding dimension: {len(response['embedding'])}")
        return True
        
    except Exception as e:
        print(f"❌ Gemini connection failed: {e}")
        return False

def test_pinecone_connection():
    """Test Pinecone connection for Gemini index"""
    config = get_gemini_config()
    if not config:
        return False
    
    try:
        from pinecone import Pinecone
        print("🔗 Testing Pinecone connection for Gemini...")
        
        pc = Pinecone(api_key=config['pinecone_api_key'])
        indexes = pc.list_indexes()
        index_names = [idx.name for idx in indexes.indexes] if hasattr(indexes, 'indexes') else []
        
        print(f"✅ Pinecone connection successful!")
        print(f"Available indexes: {index_names}")
        
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

def save_gemini_embeddings(chunks_with_embeddings, filename="gemini_embeddings_metadata.json"):
    """Save Gemini embeddings metadata to local JSON file"""
    # Remove embeddings to reduce file size
    save_data = []
    for chunk in chunks_with_embeddings:
        save_chunk = {k: v for k, v in chunk.items() if k != 'embedding'}
        save_chunk['has_embedding'] = True
        save_data.append(save_chunk)
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(save_data)} Gemini embeddings to {filename}")

def load_chunks_from_txt_files(directory=".", pattern="chunk_*.txt"):
    """Load chunks from individual text files"""
    chunk_files = glob.glob(os.path.join(directory, pattern))
    chunk_files.sort()
    
    chunks = []
    for file_path in chunk_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            
            if text:
                chunk_data = {
                    'text': text,
                    'word_count': len(text.split()),
                    'char_count': len(text),
                    'file_name': os.path.basename(file_path)
                }
                chunks.append(chunk_data)
                
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    print(f"Loaded {len(chunks)} chunks from {len(chunk_files)} files")
    return chunks

# Usage estimation
def estimate_gemini_usage(chunks):
    """Estimate Gemini embedding usage"""
    total_chars = sum(len(chunk['text']) for chunk in chunks)
    
    print(f"\n📊 Usage Estimation:")
    print(f"   Total characters: {total_chars}")
    print(f"   Number of chunks: {len(chunks)}")
    print(f"   Note: Gemini embeddings are currently free with usage limits")

# Main execution function
def main():
    """Main function to run Gemini embedding pipeline"""

    print("=== GEMINI EMBEDDING PIPELINE ===")

    # Test connections first
    print("1. Testing connections...")
    if not test_gemini_connection():
        print("Cannot continue without Gemini connection")
        return
    
    if not test_pinecone_connection():
        print("Cannot continue without Pinecone connection")
        return
    
    # Get configuration
    config = get_gemini_config()
    if not config:
        return
    
    # Load chunks
    print("\n2. Loading chunks...")
    chunks = load_chunks_from_txt_files(pattern="chunk_*.txt")
    
    if not chunks:
        print("No chunks found to process.")
        return
    
    print(f"Processing {len(chunks)} chunks...")
    
    # Estimate usage
    estimate_gemini_usage(chunks)

    # Initialize Gemini embedder
    embedder = GeminiEmbedder(
        google_api_key=config['gemini_api_key'],
        model=config['model'],
        pinecone_api_key=config['pinecone_api_key'],
        index_name=config['index_name']
    )
    
    # Process chunks with Gemini
    print("\n3. Generating embeddings with Gemini...")
    chunks_with_embeddings = embedder.process_chunks(chunks, store_in_pinecone=True)
    
    # Save embeddings locally
    print("\n4. Saving embeddings metadata locally...")
    save_gemini_embeddings(chunks_with_embeddings)

    print("\n✅ GEMINI EMBEDDING GENERATION COMPLETE")
    print(f"   Processed {len(chunks_with_embeddings)} chunks")
    
    if chunks_with_embeddings:
        print(f"   Embedding dimension: {chunks_with_embeddings[0]['embedding_dim']}")
        print(f"   Model used: {chunks_with_embeddings[0]['embedding_model']}")

if __name__ == "__main__":
    # Install required packages first
    print("Required packages: pip install google-generativeai pinecone")
    
    # Run the OpenAI pipeline
    main()