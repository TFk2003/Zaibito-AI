import re
import nltk
import google.generativeai as genai
from typing import Dict, List, Any
from datetime import datetime

class GeminiQueryProcessor:
    def __init__(self, google_api_key=None, model="gemini-embedding-001"):
        self.google_api_key = google_api_key
        self.model = model
        self.client = None
        self.stop_words = None
        
    def initialize_gemini(self):
        """Initialize Gemini client"""
        try:
            print("Initializing Gemini client for query processing...")
            
            if not self.google_api_key:
                raise ValueError("No Google API key provided")
            
            genai.configure(api_key=self.google_api_key)
            self.client = genai
            print("Gemini client initialized successfully!")
            
        except ImportError:
            raise ImportError("google-generativeai not installed. Run: pip install google-generativeai")
        except Exception as e:
            raise Exception(f"Error initializing Gemini: {e}")
    
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
                'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", 
                "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 
                'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 
                'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 
                'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 
                'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 
                'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 
                'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 
                'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 
                'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 
                'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 
                'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 
                'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 
                'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 
                're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 
                'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', 
                "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', 
                "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 
                'wouldn', "wouldn't"
            }
            print("Using basic stopwords set")
    
    def clean_query(self, query: str) -> str:
        """Clean and preprocess the user query"""
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
        """Remove stopwords from the query"""
        if not self.stop_words:
            self.load_stopwords()
        
        words = query.split()
        filtered_words = [word for word in words if word not in self.stop_words]
        result = ' '.join(filtered_words)
        
        print(f"After stopword removal: '{result}'")
        return result
    
    def detect_intent(self, query: str) -> Dict[str, Any]:
        """Detect the intent of the query based on keywords"""
        query_lower = query.lower()
        intent_keywords = {
            'property_registration': [
                'register property', 'property registration', 'land registration', 
                'property deed', 'property document', 'title deed', 'property record',
                'transfer property', 'property transfer'
            ],
            'lease_drafting': [
                'draft lease', 'lease agreement', 'rental agreement', 'lease document',
                'tenant agreement', 'rent contract', 'rental contract'
            ],
            'legal_advice': [
                'legal advice', 'lawyer', 'attorney', 'legal help', 'legal consultation',
                'what should i do', 'is it legal', 'legal opinion'
            ],
            'document_requirements': [
                'documents needed', 'required documents', 'what documents', 
                'papers required', 'documentation', 'documents required'
            ],
            'procedure_inquiry': [
                'how to', 'procedure', 'process', 'steps', 'what is the process',
                'how do i', 'how can i', 'what are the steps'
            ],
            'property_laws': [
                'property laws', 'real estate laws', 'land laws', 'property act',
                'legal framework', 'property regulations'
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
        
        # Handle empty confidence_scores properly
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
        """Extract key phrases from the query for better search"""
        words = query.split()
        key_phrases = []
        
        # Look for consecutive nouns or important terms
        legal_terms = {
            'property', 'lease', 'agreement', 'contract', 'document', 'registration', 
            'deed', 'title', 'land', 'real estate', 'karachi', 'pakistan', 'legal', 
            'law', 'act', 'regulation', 'transfer', 'ownership', 'tenant', 'landlord',
            'rent', 'rental', 'draft', 'procedure', 'requirement', 'documentation'
        }
        
        current_phrase = []
        for word in words:
            # Remove punctuation from word for comparison
            clean_word = word.strip('?.!,"')
            if clean_word in legal_terms or len(clean_word) > 4:
                current_phrase.append(word)
            else:
                if len(current_phrase) >= 1:
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
        """Convert query to vector using Gemini API"""
        if not self.client:
            self.initialize_gemini()
        
        print("Converting query to vector using Gemini...")
        
        try:
            response = genai.embed_content(
                model=self.model,
                content=query
            )
            
            vector = response['embedding']
            
            print(f"Gemini vector generated - Dimension: {len(vector)}")
            print(f"First 5 values: {vector[:5]}")
            
            return vector
            
        except Exception as e:
            print(f"❌ Error converting query to vector: {e}")
            raise
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Complete query preprocessing pipeline for Gemini"""
        print("=== GEMINI QUERY PREPROCESSING PIPELINE ===")
        
        # Step 1: Clean the query
        cleaned_query = self.clean_query(query)
        
        # Step 2: Remove stopwords (optional)
        query_without_stopwords = self.remove_stopwords(cleaned_query)
        
        # Step 3: Detect intent
        intent_analysis = self.detect_intent(cleaned_query)
        
        # Step 4: Extract key phrases
        key_phrases = self.extract_key_phrases(cleaned_query)
        
        # Step 5: Convert to vector using Gemini API
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
            'embedding_model': self.model,
            'is_legal_related': intent_analysis['is_legal_query']
        }
        
        print("\n✅ GEMINI QUERY PREPROCESSING COMPLETE")
        return result

# Configuration for Gemini
def get_gemini_query_config():
    """Get Gemini configuration for query processing"""
    GOOGLE_API_KEY = "AIzaSyCohN1h7KxM3iaUuO-Eg_gUbUYBUQ7eGZw"
    MODEL = "gemini-embedding-001"
    
    if not GOOGLE_API_KEY or len(GOOGLE_API_KEY) < 20:
        print("❌ INVALID GOOGLE API KEY")
        return None
    
    print("✅ Valid Google API key found for query processing")
    print(f"📊 Using model: {MODEL}")
    
    return {
        'google_api_key': GOOGLE_API_KEY,
        'model': MODEL
    }

# Test function for Gemini queries
def test_gemini_queries():
    """Test the Gemini query processor with sample queries"""
    config = get_gemini_query_config()
    if not config:
        return
    
    processor = GeminiQueryProcessor(
        google_api_key=config['google_api_key'],
        model=config['model']
    )
    
    test_queries = [
        "How do I register a property in Karachi?",
        "I need to draft a lease agreement for my apartment",
        "What documents are required for property registration?",
        "Tell me about property transfer laws in Pakistan",
        "How to create a rental contract for commercial property?"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Testing Gemini query: '{query}'")
        print('='*60)
        
        result = processor.process_query(query)
        
        print(f"\n📋 Gemini Results:")
        print(f"  Cleaned: {result['cleaned_query']}")
        print(f"  Without stopwords: {result['query_without_stopwords']}")
        print(f"  Primary intent: {result['intent_analysis']['primary_intent']}")
        print(f"  Confidence: {result['intent_analysis']['confidence_scores'].get(result['intent_analysis']['primary_intent'], 0):.2f}")
        print(f"  Key phrases: {result['key_phrases']}")
        print(f"  Vector dimension: {result['vector_dimension']}")
        print(f"  Model: {result['embedding_model']}")
        print(f"  Legal related: {result['is_legal_related']}")

# Main execution for Gemini queries
def main():
    """Main function to run Gemini query preprocessing"""
    config = get_gemini_query_config()
    if not config:
        print("Cannot start without valid Gemini configuration")
        return
    
    processor = GeminiQueryProcessor(
        google_api_key=config['google_api_key'],
        model=config['model']
    )
    
    while True:
        print("\n" + "="*60)
        print("GEMINI QUERY PREPROCESSING TOOL")
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
            
            print(f"\n🎯 GEMINI FINAL RESULTS:")
            print(f"  Original: {result['original_query']}")
            print(f"  Cleaned: {result['cleaned_query']}")
            print(f"  Intent: {result['intent_analysis']['primary_intent']}")
            print(f"  Confidence: {result['intent_analysis']['confidence_scores'].get(result['intent_analysis']['primary_intent'], 0):.2f}")
            print(f"  Key phrases: {', '.join(result['key_phrases'])}")
            print(f"  Vector ready: ✓ ({result['vector_dimension']} dimensions)")
            print(f"  Model: {result['embedding_model']}")
            
        except Exception as e:
            print(f"❌ Error processing query with Gemini: {e}")

if __name__ == "__main__":
    # Install required packages first
    print("Required packages: pip install google-generativeai nltk")
    
    # Run test queries first
    print("🧪 Running Gemini test queries...")
    test_gemini_queries()
    
    # Then run interactive mode
    print("\n\n🎮 Starting Gemini interactive mode...")
    main()