import json
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

class DocumentRetriever:
    def __init__(self, pinecone_api_key=None, index_name="", model_name='all-MiniLM-L6-v2'):
        self.pinecone_api_key = pinecone_api_key
        self.index_name = index_name
        self.model_name = model_name
        self.model = None
        self.index = None
        
    def load_model(self):
        """Load the sentence transformer model"""
        try:
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
            from pinecone import Pinecone
            print("Initializing Pinecone for retrieval...")
            
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
    
    def retrieve_from_pinecone(self, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve top_k documents from Pinecone using vector similarity search
        
        Args:
            query_vector: The query embedding vector
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
                    'chunk_index': match['metadata'].get('chunk_index', 'unknown'),
                    'vector_id': match['id']
                }
                retrieved_docs.append(doc_data)
                print(f"  #{i+1} - Score: {match['score']:.4f} - {match['metadata'].get('file_name', 'unknown')}")
            
            print(f"✅ Retrieved {len(retrieved_docs)} documents from Pinecone")
            return retrieved_docs
            
        except Exception as e:
            print(f"❌ Error querying Pinecone: {e}")
            return []
    
    def retrieve_from_local(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve documents from local JSON file using simple text matching
        Fallback method if Pinecone is unavailable
        """
        try:
            print("🔍 Searching local embeddings metadata...")
            
            # Load local embeddings metadata
            with open('embeddings_metadata.json', 'r', encoding='utf-8') as f:
                chunks_data = json.load(f)
            
            # Simple keyword-based matching (fallback)
            query_lower = query_text.lower()
            scored_chunks = []
            
            for chunk in chunks_data:
                score = 0
                text_lower = chunk.get('text', '').lower()
                file_name = chunk.get('file_name', 'unknown')
                
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
                        'source': 'local_fallback'
                    })
            
            # Sort by score and take top_k
            scored_chunks.sort(key=lambda x: x['score'], reverse=True)
            top_chunks = scored_chunks[:top_k]
            
            # Add rank
            for i, chunk in enumerate(top_chunks):
                chunk['rank'] = i + 1
            
            print(f"✅ Retrieved {len(top_chunks)} documents from local storage")
            return top_chunks
            
        except Exception as e:
            print(f"❌ Error retrieving from local storage: {e}")
            return []
    
    def process_query(self, query_text: str, top_k: int = 5, use_pinecone: bool = True) -> Dict[str, Any]:
        """
        Complete document retrieval pipeline
        
        Args:
            query_text: The user's query
            top_k: Number of results to return
            use_pinecone: Whether to use Pinecone (True) or local fallback (False)
            
        Returns:
            Dictionary containing retrieval results
        """
        print("=== DOCUMENT RETRIEVAL PIPELINE ===")
        print(f"Query: '{query_text}'")
        print(f"Top K: {top_k}")
        print(f"Using Pinecone: {use_pinecone}")
        
        # Load model for vector conversion if using Pinecone
        if use_pinecone and not self.model:
            self.load_model()
        
        retrieval_results = {}
        
        if use_pinecone and self.pinecone_api_key:
            try:
                # Convert query to vector
                print("Converting query to vector...")
                query_vector = self.model.encode([query_text])[0].tolist()
                print(f"Query vector dimension: {len(query_vector)}")
                
                # Retrieve from Pinecone
                retrieved_docs = self.retrieve_from_pinecone(query_vector, top_k)
                retrieval_results = {
                    'retrieved_documents': retrieved_docs,
                    'total_retrieved': len(retrieved_docs),
                    'retrieval_method': 'pinecone_semantic',
                    'query_vector_used': True
                }
                
            except Exception as e:
                print(f"⚠️ Pinecone retrieval failed, falling back to local: {e}")
                # Fallback to local retrieval
                retrieved_docs = self.retrieve_from_local(query_text, top_k)
                retrieval_results = {
                    'retrieved_documents': retrieved_docs,
                    'total_retrieved': len(retrieved_docs),
                    'retrieval_method': 'local_keyword_fallback',
                    'query_vector_used': False
                }
        else:
            # Use local retrieval
            retrieved_docs = self.retrieve_from_local(query_text, top_k)
            retrieval_results = {
                'retrieved_documents': retrieved_docs,
                'total_retrieved': len(retrieved_docs),
                'retrieval_method': 'local_keyword',
                'query_vector_used': False
            }
        
        # Add query info to results
        retrieval_results['query'] = query_text
        retrieval_results['top_k_requested'] = top_k
        
        print(f"\n✅ RETRIEVAL COMPLETE")
        print(f"   Method: {retrieval_results['retrieval_method']}")
        print(f"   Documents found: {retrieval_results['total_retrieved']}")
        
        return retrieval_results

# Configuration
def get_retriever_config():
    """Get retriever configuration"""
    PINECONE_API_KEY = "your-pinecone-api-key"
    INDEX_NAME = "your-index-name"
    MODEL_NAME = "all-MiniLM-L6-v2"
    
    return {
        'pinecone_api_key': PINECONE_API_KEY,
        'index_name': INDEX_NAME,
        'model_name': MODEL_NAME
    }

# Utility functions
def display_results(results: Dict[str, Any]):
    """Display retrieval results in a formatted way"""
    print(f"\n📊 RETRIEVAL RESULTS")
    print(f"Query: '{results['query']}'")
    print(f"Method: {results['retrieval_method']}")
    print(f"Documents found: {results['total_retrieved']}/{results['top_k_requested']}")
    print("-" * 80)
    
    for doc in results['retrieved_documents']:
        print(f"\n🏆 Rank #{doc['rank']}")
        print(f"   Score: {doc.get('score', 0):.4f}")
        print(f"   Source: {doc.get('file_name', 'unknown')}")
        print(f"   Words: {doc.get('word_count', 0)}")
        print(f"   Preview: {doc['text'][:150]}...")
        print("-" * 60)

def save_results(results: Dict[str, Any], filename: str = "retrieval_results.json"):
    """Save retrieval results to JSON file"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"💾 Results saved to: {filename}")
    except Exception as e:
        print(f"❌ Error saving results: {e}")

# Test function
def test_retrieval():
    """Test the document retriever with sample queries"""
    config = get_retriever_config()
    retriever = DocumentRetriever(**config)
    
    test_queries = [
        "property registration process",
        "lease agreement requirements", 
        "legal documents needed",
        "how to transfer property",
        "tenant rights and responsibilities"
    ]
    
    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"TESTING RETRIEVAL: '{query}'")
        print('='*80)
        
        results = retriever.process_query(query, top_k=3, use_pinecone=True)
        display_results(results)
        
        # Save first test results
        if query == test_queries[0]:
            save_results(results, "test_retrieval_results.json")

# Main interactive function
def main():
    """Main interactive document retrieval interface"""
    config = get_retriever_config()
    retriever = DocumentRetriever(**config)
    
    print("🔍 DOCUMENT RETRIEVAL SYSTEM")
    print("=" * 50)
    
    while True:
        print("\nOptions:")
        print("1. Search with Pinecone (semantic search)")
        print("2. Search locally (keyword fallback)")
        print("3. Test with sample queries")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '4':
            print("Goodbye!")
            break
            
        elif choice == '3':
            test_retrieval()
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
                
                display_results(results)
                
                # Ask to save results
                save_choice = input("\nSave results to file? (y/n): ").strip().lower()
                if save_choice == 'y':
                    filename = input("Filename (default: retrieval_results.json): ").strip() or "retrieval_results.json"
                    save_results(results, filename)
                    
            except ValueError:
                print("Please enter a valid number for top_k.")
            except Exception as e:
                print(f"Error during retrieval: {e}")
                
        else:
            print("Invalid choice. Please enter 1-4.")

if __name__ == "__main__":
    # Install required packages
    print("Required packages: pip install sentence-transformers pinecone")
    
    # Run test first
    print("🧪 Running retrieval tests...")
    test_retrieval()
    
    # Then run interactive mode
    print("\n\n🎮 Starting interactive retrieval system...")
    main()