import re
import nltk
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Any

class QueryProcessor:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.model = None
        self.stop_words = None
        
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
    
    def load_stopwords(self):
        """Load stopwords for query cleaning"""
        try:
            nltk.download('stopwords', quiet=True)
            from nltk.corpus import stopwords
            self.stop_words = set(stopwords.words('english'))
            print("Stopwords loaded successfully!")
        except:
            # Fallback basic stopwords
            self.stop_words = {
                'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", 
                "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 
                'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 
                'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 
                'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 
                'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 
                'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 
                'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 
                'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 
                'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 
                'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 
                'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 
                'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 
                'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 
                'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', 
                "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 
                'wouldn', "wouldn't"
            }
            print("Using basic stopwords set")
    
    def clean_query(self, query: str) -> str:
        """
        Clean and preprocess the user query
        """
        if not query:
            return ""
        
        print(f"Original query: '{query}'")
        
        # Convert to lowercase
        cleaned = query.lower()
        
        # Remove special characters but keep basic punctuation and numbers
        cleaned = re.sub(r'[^\w\s\.\?]', '', cleaned)
        
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        print(f"After cleaning: '{cleaned}'")
        return cleaned
    
    def remove_stopwords(self, query: str) -> str:
        """
        Remove stopwords from the query
        """
        if not self.stop_words:
            self.load_stopwords()
        
        words = query.split()
        filtered_words = [word for word in words if word not in self.stop_words]
        result = ' '.join(filtered_words)
        
        print(f"After stopword removal: '{result}'")
        return result
    
    def detect_intent(self, query: str) -> Dict[str, Any]:
        """
        Detect the intent of the query based on keywords
        """
        query_lower = query.lower()
        intent_keywords = {
            'property_registration': [
                'register property', 'property registration', 'land registration', 
                'property deed', 'property document', 'title deed', 'property record'
            ],
            'lease_drafting': [
                'draft lease', 'lease agreement', 'rental agreement', 'lease document',
                'tenant agreement', 'rent contract'
            ],
            'legal_advice': [
                'legal advice', 'lawyer', 'attorney', 'legal help', 'legal consultation',
                'what should i do', 'is it legal'
            ],
            'document_requirements': [
                'documents needed', 'required documents', 'what documents', 
                'papers required', 'documentation'
            ],
            'procedure_inquiry': [
                'how to', 'procedure', 'process', 'steps', 'what is the process',
                'how do i', 'how can i'
            ]
        }
        
        detected_intents = []
        confidence_scores = {}
        
        for intent, keywords in intent_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in query_lower:
                    score += 1
            
            if score > 0:
                confidence = min(score / len(keywords), 1.0)
                confidence_scores[intent] = confidence
                detected_intents.append(intent)
        
        # FIXED: Handle empty confidence_scores properly
        if confidence_scores:
            primary_intent = max(confidence_scores, key=lambda k: confidence_scores[k])
        else:
            primary_intent = 'general_inquiry'
            confidence_scores['general_inquiry'] = 0.0
        
        intent_result = {
            'primary_intent': primary_intent,
            'all_intents': detected_intents,
            'confidence_scores': confidence_scores,
            'is_legal_query': len(detected_intents) > 0
        }
        
        print(f"Detected intents: {intent_result}")
        return intent_result
    
    def extract_key_phrases(self, query: str) -> List[str]:
        """
        Extract key phrases from the query for better search
        """
        # Simple noun phrase extraction (can be enhanced with NER)
        words = query.split()
        key_phrases = []
        
        # Look for consecutive nouns or important terms
        legal_terms = {'property', 'lease', 'agreement', 'contract', 'document', 
                      'registration', 'deed', 'title', 'land', 'real estate', 'karachi',
                      'pakistan', 'legal', 'law'}
        
        current_phrase = []
        for word in words:
            # Remove punctuation from word for comparison
            clean_word = word.strip('?.!,"')
            if clean_word in legal_terms or len(clean_word) > 4:  # Assume longer words are more important
                current_phrase.append(word)
            else:
                if len(current_phrase) >= 1:  # Reduced to 1 to catch single important words
                    key_phrases.append(' '.join(current_phrase))
                current_phrase = []
        
        # Add any remaining phrase
        if len(current_phrase) >= 1:
            key_phrases.append(' '.join(current_phrase))
        
        # Also include the full cleaned query as a key phrase
        if query and query not in key_phrases:
            key_phrases.append(query)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_phrases = []
        for phrase in key_phrases:
            if phrase not in seen:
                seen.add(phrase)
                unique_phrases.append(phrase)
        
        print(f"Extracted key phrases: {unique_phrases}")
        return unique_phrases
    
    def convert_to_vector(self, query: str):
        """
        Convert query to vector using the same embedding model
        """
        if not self.model:
            self.load_model()
        
        print("Converting query to vector...")
        vector = self.model.encode([query])[0].tolist()
        
        print(f"Vector generated - Dimension: {len(vector)}")
        print(f"First 5 values: {vector[:5]}")
        
        return vector
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Complete query preprocessing pipeline
        """
        print("=== QUERY PREPROCESSING PIPELINE ===")
        
        # Step 1: Clean the query
        cleaned_query = self.clean_query(query)
        
        # Step 2: Remove stopwords (optional - can be skipped for better semantic understanding)
        query_without_stopwords = self.remove_stopwords(cleaned_query)
        
        # Step 3: Detect intent
        intent_analysis = self.detect_intent(cleaned_query)
        
        # Step 4: Extract key phrases
        key_phrases = self.extract_key_phrases(cleaned_query)
        
        # Step 5: Convert to vector (use original cleaned query for better semantics)
        query_vector = self.convert_to_vector(cleaned_query)
        
        # Prepare result
        result = {
            'original_query': query,
            'cleaned_query': cleaned_query,
            'query_without_stopwords': query_without_stopwords,
            'intent_analysis': intent_analysis,
            'key_phrases': key_phrases,
            'query_vector': query_vector,
            'vector_dimension': len(query_vector),
            'is_legal_related': intent_analysis['is_legal_query']
        }
        
        print("\n✅ QUERY PREPROCESSING COMPLETE")
        return result

# Utility functions
def test_queries():
    """Test the query processor with sample queries"""
    processor = QueryProcessor()
    
    test_queries = [
        "How do I register a property in Karachi?",
        "I need to draft a lease agreement for my apartment",
        "What documents are required for property registration?",
        "Is this legal procedure correct?",
        "Tell me about property laws in Pakistan",
        "Hello world"  # Test with non-legal query
    ]
    
    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"Testing query: '{query}'")
        print('='*50)
        
        result = processor.process_query(query)
        
        print(f"\n📋 Results:")
        print(f"  Cleaned: {result['cleaned_query']}")
        print(f"  Without stopwords: {result['query_without_stopwords']}")
        print(f"  Primary intent: {result['intent_analysis']['primary_intent']}")
        print(f"  Confidence: {result['intent_analysis']['confidence_scores'].get(result['intent_analysis']['primary_intent'], 0):.2f}")
        print(f"  Key phrases: {result['key_phrases']}")
        print(f"  Vector dimension: {result['vector_dimension']}")
        print(f"  Legal related: {result['is_legal_related']}")

# Main execution
def main():
    """Main function to run query preprocessing"""
    processor = QueryProcessor()
    
    while True:
        print("\n" + "="*60)
        print("QUERY PREPROCESSING TOOL")
        print("="*60)
        print("Enter your query (or 'quit' to exit):")
        
        user_query = input("> ").strip()
        
        if user_query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not user_query:
            print("Please enter a query.")
            continue
        
        try:
            result = processor.process_query(user_query)
            
            print(f"\n🎯 FINAL RESULTS:")
            print(f"  Original: {result['original_query']}")
            print(f"  Cleaned: {result['cleaned_query']}")
            print(f"  Intent: {result['intent_analysis']['primary_intent']}")
            print(f"  Confidence: {result['intent_analysis']['confidence_scores'].get(result['intent_analysis']['primary_intent'], 0):.2f}")
            print(f"  Key phrases: {', '.join(result['key_phrases'])}")
            print(f"  Vector ready: ✓ ({result['vector_dimension']} dimensions)")
            
        except Exception as e:
            print(f"❌ Error processing query: {e}")

if __name__ == "__main__":
    # Run test queries first
    print("🧪 Running test queries...")
    test_queries()
    
    # Then run interactive mode
    print("\n\n🎮 Starting interactive mode...")
    main()