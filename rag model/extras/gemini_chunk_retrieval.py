import json
import google.generativeai as genai
from typing import List, Dict, Any

class GeminiDocumentRetriever:
    def __init__(self, google_api_key=None, pinecone_api_key=None, index_name="your-index-name", model_name="gemini-embedding-001"):
        self.google_api_key = google_api_key
        self.pinecone_api_key = pinecone_api_key
        self.index_name = index_name
        self.model_name = model_name
        self.client = None
        self.index = None
        
    def initialize_gemini(self):
        """Initialize Gemini client"""
        try:
            print("Initializing Gemini client for retrieval...")
            
            if not self.google_api_key:
                raise ValueError("No Google API key provided")
            
            genai.configure(api_key=self.google_api_key)
            self.client = genai
            print("Gemini client initialized successfully!")
            
        except ImportError:
            raise ImportError("google-generativeai not installed. Run: pip install google-generativeai")
        except Exception as e:
            raise Exception(f"Error initializing Gemini: {e}")
    
    def initialize_pinecone(self):
        """Initialize Pinecone connection for Gemini embeddings"""
        try:
            from pinecone import Pinecone
            print("Initializing Pinecone for Gemini retrieval...")
            
            if not self.pinecone_api_key:
                raise ValueError("No Pinecone API key provided")
            
            # Initialize Pinecone client
            pc = Pinecone(api_key=self.pinecone_api_key)
            
            # Connect to index
            self.index = pc.Index(self.index_name)
            print(f"Connected to Pinecone index: {self.index_name}")
            
            # Print index stats
            index_stats = self.index.describe_index_stats()
            print(f"Index stats - Vectors: {index_stats['total_vector_count']}, Dimension: {index_stats['dimension']}")
            
        except ImportError:
            raise ImportError("pinecone not installed. Run: pip install pinecone")
        except Exception as e:
            raise Exception(f"Error initializing Pinecone: {e}")
    
    def convert_query_to_vector(self, query_text: str) -> List[float]:
        """Convert query to vector using Gemini API"""
        if not self.client:
            self.initialize_gemini()
        
        print("Converting query to vector using Gemini...")
        
        try:
            response = genai.embed_content(
                model=self.model_name,
                content=query_text
            )
            
            vector = response['embedding']
            print(f"Gemini vector generated - Dimension: {len(vector)}")
            return vector
            
        except Exception as e:
            print(f"❌ Error converting query to vector with Gemini: {e}")
            raise
    
    def retrieve_from_pinecone(self, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve top_k documents from Pinecone using vector similarity search
        
        Args:
            query_vector: The query embedding vector (3072 dimensions)
            top_k: Number of top results to return
            
        Returns:
            List of retrieved documents with metadata
        """
        if not self.index:
            self.initialize_pinecone()
        
        print(f"🔍 Searching Pinecone for top {top_k} matches...")
        
        try:
            # Query Pinecone index
            results = self.index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True,
                include_values=False  # We don't need the vector values, just metadata
            )
            
            retrieved_docs = []
            for i, match in enumerate(results['matches']):
                doc_data = {
                    'rank': i + 1,
                    'score': match['score'],
                    'text': match['metadata'].get('text', ''),
                    'file_name': match['metadata'].get('file_name', 'unknown'),
                    'word_count': match['metadata'].get('word_count', 0),
                    'char_count': match['metadata'].get('char_count', 0),
                    'embedding_model': match['metadata'].get('embedding_model', 'unknown'),
                    'chunk_index': match['metadata'].get('chunk_index', 'unknown'),
                    'vector_id': match['id'],
                    'retrieval_method': 'gemini_pinecone_semantic'
                }
                retrieved_docs.append(doc_data)
                print(f"  #{i+1} - Score: {match['score']:.4f} - {match['metadata'].get('file_name', 'unknown')}")
            
            print(f"✅ Retrieved {len(retrieved_docs)} documents from Pinecone using Gemini embeddings")
            return retrieved_docs
            
        except Exception as e:
            print(f"❌ Error querying Pinecone: {e}")
            return []
    
    def retrieve_from_local_gemini(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve documents from local Gemini embeddings JSON file using keyword matching
        Fallback method if Pinecone is unavailable
        """
        try:
            print("🔍 Searching local Gemini embeddings metadata...")
            
            # Load local Gemini embeddings metadata
            with open('gemini_embeddings_metadata.json', 'r', encoding='utf-8') as f:
                chunks_data = json.load(f)
            
            # Simple keyword-based matching (fallback)
            query_lower = query_text.lower()
            scored_chunks = []
            
            for chunk in chunks_data:
                score = 0
                text_lower = chunk.get('text', '').lower()
                file_name = chunk.get('file_name', 'unknown')
                embedding_model = chunk.get('embedding_model', 'unknown')
                
                # Only consider chunks that were embedded with Gemini
                if embedding_model != self.model_name:
                    continue
                
                # Simple scoring based on keyword matches
                for word in query_lower.split():
                    if len(word) > 3:  # Only consider words longer than 3 characters
                        score += text_lower.count(word) * 2
                
                # Bonus for exact phrase matches
                if query_lower in text_lower:
                    score += 10
                
                if score > 0:
                    scored_chunks.append({
                        'score': score,
                        'text': chunk.get('text', ''),
                        'file_name': file_name,
                        'word_count': chunk.get('word_count', 0),
                        'char_count': chunk.get('char_count', 0),
                        'embedding_model': embedding_model,
                        'source': 'local_gemini_fallback',
                        'retrieval_method': 'gemini_local_keyword'
                    })
            
            # Sort by score and take top_k
            scored_chunks.sort(key=lambda x: x['score'], reverse=True)
            top_chunks = scored_chunks[:top_k]
            
            # Add rank
            for i, chunk in enumerate(top_chunks):
                chunk['rank'] = i + 1
            
            print(f"✅ Retrieved {len(top_chunks)} documents from local Gemini storage")
            return top_chunks
            
        except Exception as e:
            print(f"❌ Error retrieving from local Gemini storage: {e}")
            return []
    
    def hybrid_retrieval(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval that tries Pinecone first, falls back to local if needed
        """
        print("🔄 Attempting hybrid retrieval...")
        
        # Try Pinecone first
        if self.pinecone_api_key:
            try:
                query_vector = self.convert_query_to_vector(query_text)
                pinecone_results = self.retrieve_from_pinecone(query_vector, top_k)
                if pinecone_results:
                    print("✅ Using Pinecone semantic search results")
                    return pinecone_results
            except Exception as e:
                print(f"⚠️ Pinecone retrieval failed: {e}")
        
        # Fallback to local Gemini storage
        print("🔄 Falling back to local Gemini storage...")
        local_results = self.retrieve_from_local_gemini(query_text, top_k)
        return local_results
    
    def process_query(self, query_text: str, top_k: int = 5, use_pinecone: bool = True) -> Dict[str, Any]:
        """
        Complete document retrieval pipeline for Gemini embeddings
        
        Args:
            query_text: The user's query
            top_k: Number of results to return
            use_pinecone: Whether to use Pinecone (True) or local fallback (False)
            
        Returns:
            Dictionary containing retrieval results
        """
        print("=== GEMINI DOCUMENT RETRIEVAL PIPELINE ===")
        print(f"Query: '{query_text}'")
        print(f"Top K: {top_k}")
        print(f"Using Pinecone: {use_pinecone}")
        print(f"Embedding Model: {self.model_name}")
        
        retrieval_results = {}
        
        if use_pinecone:
            # Use hybrid approach (Pinecone with local fallback)
            retrieved_docs = self.hybrid_retrieval(query_text, top_k)
            retrieval_method = retrieved_docs[0]['retrieval_method'] if retrieved_docs else 'no_results'
        else:
            # Use local only
            retrieved_docs = self.retrieve_from_local_gemini(query_text, top_k)
            retrieval_method = 'gemini_local_keyword'
        
        retrieval_results = {
            'retrieved_documents': retrieved_docs,
            'total_retrieved': len(retrieved_docs),
            'retrieval_method': retrieval_method,
            'embedding_model': self.model_name,
            'query_vector_used': use_pinecone,
            'vector_dimension': 3072  # Gemini embeddings are 3072D
        }
        
        # Add query info to results
        retrieval_results['query'] = query_text
        retrieval_results['top_k_requested'] = top_k
        
        print(f"\n✅ GEMINI RETRIEVAL COMPLETE")
        print(f"   Method: {retrieval_results['retrieval_method']}")
        print(f"   Model: {retrieval_results['embedding_model']}")
        print(f"   Documents found: {retrieval_results['total_retrieved']}")
        
        return retrieval_results

# Configuration for Gemini
def get_gemini_retriever_config():
    """Get Gemini retriever configuration"""
    GOOGLE_API_KEY = "your-google-api-key"
    PINECONE_API_KEY = "your-pinecone-api-key"
    INDEX_NAME = "your-index-name"
    MODEL_NAME = "gemini-embedding-001"
    
    return {
        'google_api_key': GOOGLE_API_KEY,
        'pinecone_api_key': PINECONE_API_KEY,
        'index_name': INDEX_NAME,
        'model_name': MODEL_NAME
    }

# Utility functions
def display_gemini_results(results: Dict[str, Any]):
    """Display Gemini retrieval results in a formatted way"""
    print(f"\n📊 GEMINI RETRIEVAL RESULTS")
    print(f"Query: '{results['query']}'")
    print(f"Method: {results['retrieval_method']}")
    print(f"Model: {results['embedding_model']}")
    print(f"Vector Dimension: {results['vector_dimension']}")
    print(f"Documents found: {results['total_retrieved']}/{results['top_k_requested']}")
    print("-" * 80)
    
    for doc in results['retrieved_documents']:
        print(f"\n🏆 Rank #{doc['rank']}")
        print(f"   Score: {doc.get('score', 0):.4f}")
        print(f"   Source: {doc.get('file_name', 'unknown')}")
        print(f"   Embedding Model: {doc.get('embedding_model', 'unknown')}")
        print(f"   Words: {doc.get('word_count', 0)}")
        print(f"   Characters: {doc.get('char_count', 0)}")
        print(f"   Preview: {doc['text'][:150]}...")
        print("-" * 60)

def save_gemini_results(results: Dict[str, Any], filename: str = "gemini_retrieval_results.json"):
    """Save Gemini retrieval results to JSON file"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"💾 Gemini results saved to: {filename}")
    except Exception as e:
        print(f"❌ Error saving Gemini results: {e}")

# Test function for Gemini
def test_gemini_retrieval():
    """Test the Gemini document retriever with sample queries"""
    config = get_gemini_retriever_config()
    retriever = GeminiDocumentRetriever(**config)
    
    test_queries = [
        "property registration process in Karachi",
        "lease agreement drafting requirements",
        "legal documents for property transfer",
        "commercial property rental contracts",
        "real estate laws and regulations"
    ]
    
    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"TESTING GEMINI RETRIEVAL: '{query}'")
        print('='*80)
        
        results = retriever.process_query(query, top_k=3, use_pinecone=True)
        display_gemini_results(results)
        
        # Save first test results
        if query == test_queries[0]:
            save_gemini_results(results, "test_gemini_retrieval_results.json")

# Compare both retrievers
def compare_retrieval_methods():
    """Compare local Sentence Transformer vs Gemini retrieval"""
    from chunk_retreival import DocumentRetriever, get_retriever_config, display_results
    
    query = "property registration requirements"
    
    print("🔄 COMPARING RETRIEVAL METHODS")
    print("=" * 60)
    
    # Test local Sentence Transformer
    print("\n🧠 LOCAL SENTENCE TRANSFORMER:")
    local_config = get_retriever_config()
    local_retriever = DocumentRetriever(**local_config)
    local_results = local_retriever.process_query(query, top_k=3, use_pinecone=True)
    display_results(local_results)
    
    # Test Gemini
    print("\n🌟 GEMINI EMBEDDINGS:")
    gemini_config = get_gemini_retriever_config()
    gemini_retriever = GeminiDocumentRetriever(**gemini_config)
    gemini_results = gemini_retriever.process_query(query, top_k=3, use_pinecone=True)
    display_gemini_results(gemini_results)

# Main interactive function for Gemini
def main():
    """Main interactive Gemini document retrieval interface"""
    config = get_gemini_retriever_config()
    retriever = GeminiDocumentRetriever(**config)
    
    print("🔍 GEMINI DOCUMENT RETRIEVAL SYSTEM")
    print("=" * 50)
    print(f"Using model: {config['model_name']}")
    print(f"Pinecone index: {config['index_name']}")
    
    while True:
        print("\nOptions:")
        print("1. Search with Gemini + Pinecone (semantic search)")
        print("2. Search Gemini local storage only (keyword fallback)")
        print("3. Test with sample queries")
        print("4. Compare with local Sentence Transformer")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '5':
            print("Goodbye!")
            break
            
        elif choice == '3':
            test_gemini_retrieval()
            continue
            
        elif choice == '4':
            compare_retrieval_methods()
            continue
            
        elif choice in ['1', '2']:
            use_pinecone = (choice == '1')
            query = input("Enter your query: ").strip()
            
            if not query:
                print("Please enter a query.")
                continue
                
            try:
                top_k = int(input("Number of results to retrieve (default 5): ") or "5")
                results = retriever.process_query(query, top_k=top_k, use_pinecone=use_pinecone)
                
                display_gemini_results(results)
                
                # Ask to save results
                save_choice = input("\nSave results to file? (y/n): ").strip().lower()
                if save_choice == 'y':
                    filename = input("Filename (default: gemini_retrieval_results.json): ").strip() or "gemini_retrieval_results.json"
                    save_gemini_results(results, filename)
                    
            except ValueError:
                print("Please enter a valid number for top_k.")
            except Exception as e:
                print(f"Error during Gemini retrieval: {e}")
                
        else:
            print("Invalid choice. Please enter 1-5.")

if __name__ == "__main__":
    # Install required packages
    print("Required packages: pip install google-generativeai pinecone")
    
    # Run test first
    print("🧪 Running Gemini retrieval tests...")
    test_gemini_retrieval()
    
    # Then run interactive mode
    print("\n\n🎮 Starting Gemini interactive retrieval system...")
    main()