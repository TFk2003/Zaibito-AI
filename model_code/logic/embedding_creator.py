import os
import time
from model_code.core.config import settings
from model_code.controller.chunk import get_all_chunks
from typing import List, Dict
from datetime import datetime

class EmbeddingCreator:
    def __init__(self):
        self.google_api_key = settings.GOOGLE_API_KEY
        self.embedding_model = settings.EMBEDDING_MODEL
        self.pinecone_api_key = settings.PINECONE_API_KEY
        self.index_name = settings.INDEX_NAME
        self.initialize_gemini()
        self.initialize_pinecone(dimension=3072)
    
    def initialize_gemini(self):
        """Initialize Gemini client"""
        try:
            import google.generativeai as genai

            if not self.google_api_key:
                raise ValueError("No Google API key provided")
            genai.configure(api_key=self.google_api_key)
            self.client = genai
            print("Gemini client initialized successfully!")

        except ImportError:
            raise ImportError("google.generativeai not installed. Run: pip install google-generativeai")
        except Exception as e:
            raise Exception(f"Error initializing Gemini: {e}")

    def initialize_pinecone(self, dimension):
        """Initialize Pinecone connection for Gemini embeddings"""
        try:
            from pinecone import Pinecone, ServerlessSpec
            
            if not self.pinecone_api_key:
                raise ValueError("No Pinecone API key provided")
            
            # Initialize Pinecone client
            self.pc = Pinecone(api_key=self.pinecone_api_key)
            
            # Check if index exists, create if not
            existing_indexes = self.pc.list_indexes()
            index_names = [idx.name for idx in existing_indexes.indexes] if hasattr(existing_indexes, 'indexes') else []
            
            if self.index_name not in index_names:
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
        
    def get_file_chunks(self):
        all_chunks = []
        skip = 0
        limit = 100

        while True:
            chunk_page = get_all_chunks(skip=skip, limit=limit)
            if not chunk_page:
                break
            
            all_chunks.extend(chunk_page)
            skip += limit

        print(f"Total chunks fetched: {len(all_chunks)}")
        chunks = []
        for chunk in all_chunks:
            chunk_data = {
                    'text': chunk.chunk_data,
                    'word_count': len(chunk.chunk_data.split()),
                    'char_count': len(chunk.chunk_data),
                    'file_name': chunk.chunk_name
                }
            chunks.append(chunk_data)

        return chunks

    def test_connections(self):
        try:
            import google.generativeai as genai
            print("Testing Gemini connection...")

            genai.configure(api_key=self.google_api_key)

            # Test with a small request
            response = genai.embed_content(
                model=self.embedding_model,
                content="Test connection"
            )
            print(f"Gemini connection successful!")
            try:
                from pinecone import Pinecone
                print("Testing Pinecone connection for Gemini...")
                
                pc = Pinecone(api_key=self.pinecone_api_key)
                indexes = pc.list_indexes()
                index_names = [idx.name for idx in indexes.indexes] if hasattr(indexes, 'indexes') else []
                print(f"Pinecone connection successful!")
                return True
                
            except Exception as e:
                print(f"Pinecone connection failed: {e}")
                return False
        except Exception as e:
            print(f"Gemini connection failed: {e}")
            return False

    def process_chunks(self, chunks: List[Dict]):
        """
        Complete pipeline: Generate embeddings with Gemini and store in Pinecone
        """
        try:
            # Generate embeddings
            chunks_with_embeddings = self.generate_embeddings(chunks)
        
            # Store in Pinecone if requested
            self.store_in_pinecone(chunks_with_embeddings)
            return chunks_with_embeddings
        except Exception as e:
            print(f"Error in processing chunks: {e}")
            return []
    
    def generate_embeddings(self, chunks: List[Dict]) -> List[Dict]:
        """Generate embeddings using Gemini API"""
        if not self.client:
            self.initialize_gemini()
        
        chunks_with_embeddings = []
        cooldown_after = 40       # number of chunks before cooldown
        cooldown_seconds = 30
        
        for i, chunk in enumerate(chunks):
            if i > 0 and i % cooldown_after == 0:
                print(f"\n=== Cooling down for {cooldown_seconds} seconds after processing {i} chunks ===")
                time.sleep(cooldown_seconds)
            try:
                # Call Gemini API
                response = self.client.embed_content(
                    model=self.embedding_model,
                    content=chunk['text']
                )
                
                embedding = response['embedding']
                
                # Add embedding to chunk
                chunk_with_embedding = chunk.copy()
                chunk_with_embedding['embedding'] = embedding
                chunk_with_embedding['embedding_dim'] = len(embedding)
                chunk_with_embedding['embedding_model'] = self.embedding_model
                chunk_with_embedding['processed_at'] = datetime.now().isoformat()
                
                chunks_with_embeddings.append(chunk_with_embedding)
                
                # Rate limiting - be nice to the API
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error processing chunk {i+1}: {e}")
                if "429" in str(e):
                    print("Hit rate limit — waiting 60 seconds before continuing...")
                    time.sleep(60)
                # Continue with next chunk even if one fails    
                continue

        print(f"Gemini embeddings generated successfully! ({len(chunks_with_embeddings)}/{len(chunks)} chunks)")
        return chunks_with_embeddings
    
    def store_in_pinecone(self, chunks_with_embeddings: List[Dict], batch_size=50):
        """Store embeddings in Pinecone vector database"""
        # First, detect the actual embedding dimension
        if chunks_with_embeddings:
            actual_dimension = chunks_with_embeddings[0]['embedding_dim']
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
                self.index.upsert(vectors=batch)
                successful_vectors += len(batch)
                print(f"Upserted batch {i//batch_size + 1}/{(total_vectors-1)//batch_size + 1} - {len(batch)} vectors")
            except Exception as e:
                print(f"Failed to upsert batch {i//batch_size + 1}: {e}")
        
        print(f"Successfully stored {successful_vectors}/{total_vectors} vectors in Pinecone!")

    def get_embedding_dimension(self):
        """Get the actual embedding dimension by testing with a small query"""
        if not self.client:
            self.initialize_gemini()
        try:
            # Test with a small query to get actual dimension
            import google.generativeai as genai
            test_response = genai.embed_content(
                model=self.embedding_model,
                content="test"
            )
            actual_dimension = len(test_response['embedding'])
            return actual_dimension
        except Exception as e:
            print(f"Error detecting dimension, using default: {e}")
            return 3072  # Fallback dimension