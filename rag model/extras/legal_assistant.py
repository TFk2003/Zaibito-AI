import json
import os
from typing import List, Dict, Any
import google.generativeai as genai

class LegalAssistant:
    def __init__(self, google_api_key=None, model_name="gemini-2.5-flash"):
        self.google_api_key = google_api_key
        self.model_name = model_name
        self.client = None
        
    def initialize_gemini(self):
        """Initialize Gemini client for LLM"""
        try:
            print("Initializing Gemini LLM client...")
            
            if not self.google_api_key:
                raise ValueError("No Google API key provided")
            
            genai.configure(api_key=self.google_api_key)
            
            # Initialize the model
            generation_config = {
                "temperature": 0.1,  # Low temperature for factual, consistent responses
                "top_p": 0.8,
                "top_k": 40,
                "max_output_tokens": 2048,
            }
            
            safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
            ]
            
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            print(f"Gemini LLM client initialized successfully! Using model: {self.model_name}")
            
        except ImportError:
            raise ImportError("google-generativeai not installed. Run: pip install google-generativeai")
        except Exception as e:
            raise Exception(f"Error initializing Gemini LLM: {e}")
    
    def build_context_from_retrieved_docs(self, retrieved_documents: List[Dict[str, Any]]) -> str:
        """
        Build context string from retrieved documents
        """
        if not retrieved_documents:
            return "No relevant legal documents found in the knowledge base."
        
        context_parts = []
        
        for i, doc in enumerate(retrieved_documents):
            context_parts.append(f"DOCUMENT {i+1} (Relevance Score: {doc.get('score', 0):.4f}):")
            context_parts.append(f"Source: {doc.get('file_name', 'Unknown')}")
            context_parts.append(f"Content: {doc.get('text', '')}")
            context_parts.append("-" * 80)
        
        return "\n".join(context_parts)
    
    def create_legal_prompt(self, query: str, context: str, intent_analysis: Dict[str, Any] = None) -> str:
        """
        Create a specialized legal prompt for Gemini
        """
        primary_intent = intent_analysis.get('primary_intent', 'general_inquiry') if intent_analysis else 'general_inquiry'
        confidence = intent_analysis.get('confidence_scores', {}).get(primary_intent, 0) if intent_analysis else 0
        
        intent_based_instructions = {
            'property_registration': "Focus on property registration procedures, required documents, and legal steps in Sindh, Pakistan.",
            'lease_drafting': "Provide guidance on lease agreement requirements, key clauses, and legal considerations for rental contracts.",
            'legal_advice': "Offer general legal information based on the provided context. Clarify this is not formal legal advice.",
            'document_requirements': "List and explain the required documents, their purposes, and where to obtain them.",
            'procedure_inquiry': "Explain the step-by-step process, timelines, and important considerations.",
            'property_laws': "Explain relevant property laws, regulations, and legal frameworks in Pakistan.",
            'general_inquiry': "Provide helpful information based on the available legal documents."
        }
        
        intent_instruction = intent_based_instructions.get(primary_intent, intent_based_instructions['general_inquiry'])
        
        prompt = f"""
You are ZABITO, an expert legal assistant specialized in Sindh property laws and Pakistani legal procedures. 
Your role is to provide accurate, helpful information based ONLY on the provided legal documents.

**CONTEXT ANALYSIS:**
Detected User Intent: {primary_intent} (Confidence: {confidence:.2f})
{intent_instruction}

**CRITICAL INSTRUCTIONS:**
1. Answer STRICTLY based on the provided legal documents below
2. If information is not in the documents, clearly state this
3. Do not use external knowledge or make assumptions
4. Be precise, factual, and cite relevant document sections when possible
5. Structure your response clearly for legal comprehension
6. If multiple documents are relevant, synthesize information coherently

**LEGAL DOCUMENTS CONTEXT:**
{context}

**USER QUESTION:**
{query}

**RESPONSE REQUIREMENTS:**
- Start with a clear, direct answer
- Reference specific documents when applicable
- Use bullet points for procedures or requirements
- Include important caveats or limitations
- End with a summary of key points

Now provide your expert legal assistance based on the above context:
"""
        return prompt.strip()
    
    def generate_response(self, prompt: str) -> str:
        """
        Generate response using Gemini LLM
        """
        if not hasattr(self, 'model'):
            self.initialize_gemini()
        
        try:
            print("🔄 Generating response with Gemini LLM...")
            
            response = self.model.generate_content(prompt)
            
            if response.parts:
                generated_text = response.text
                print("✅ Response generated successfully!")
                return generated_text
            else:
                raise Exception("No response generated from Gemini")
                
        except Exception as e:
            error_msg = f"❌ Error generating response with Gemini: {e}"
            print(error_msg)
            return f"I apologize, but I encountered an error while processing your legal query. Please try again. Error: {str(e)}"
    
    def process_legal_query(self, query: str, retrieved_documents: List[Dict[str, Any]], 
                          intent_analysis: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Complete pipeline: Process legal query with retrieved context
        
        Args:
            query: User's legal question
            retrieved_documents: Documents from retrieval step
            intent_analysis: Intent analysis from query processing
            
        Returns:
            Dictionary containing the complete response
        """
        print("=== LEGAL ASSISTANT PIPELINE ===")
        print(f"Query: '{query}'")
        print(f"Retrieved documents: {len(retrieved_documents)}")
        
        # Step 1: Build context from retrieved documents
        context = self.build_context_from_retrieved_docs(retrieved_documents)
        print(f"Context built: {len(context)} characters")
        
        # Step 2: Create specialized legal prompt
        prompt = self.create_legal_prompt(query, context, intent_analysis)
        print(f"Prompt created: {len(prompt)} characters")
        
        # Step 3: Generate response with Gemini LLM
        response = self.generate_response(prompt)
        
        # Step 4: Prepare comprehensive results
        results = {
            'query': query,
            'response': response,
            'retrieved_documents_count': len(retrieved_documents),
            'context_character_count': len(context),
            'intent_used': intent_analysis.get('primary_intent', 'general_inquiry') if intent_analysis else 'general_inquiry',
            'model_used': self.model_name,
            'retrieved_documents_preview': [
                {
                    'rank': doc.get('rank', i+1),
                    'file_name': doc.get('file_name', 'unknown'),
                    'score': doc.get('score', 0),
                    'text_preview': doc.get('text', '')[:100] + '...' if len(doc.get('text', '')) > 100 else doc.get('text', '')
                }
                for i, doc in enumerate(retrieved_documents[:3])  # Show first 3 for preview
            ]
        }
        
        print(f"\n✅ LEGAL ASSISTANT COMPLETE")
        print(f"   Response length: {len(response)} characters")
        print(f"   Model: {self.model_name}")
        
        return results

# Configuration for Gemini Legal Assistant
def get_legal_assistant_config():
    """Get legal assistant configuration"""
    GOOGLE_API_KEY = ""
    MODEL_NAME = "gemini-2.5-flash"  # Using gemini-2.5-flash for better reasoning
    
    return {
        'google_api_key': GOOGLE_API_KEY,
        'model_name': MODEL_NAME
    }

# Utility functions
def display_legal_response(results: Dict[str, Any]):
    """Display legal assistant response in a formatted way"""
    print(f"\n🎯 LEGAL ASSISTANT RESPONSE")
    print("=" * 80)
    print(f"📝 Query: {results['query']}")
    print(f"🧠 Intent: {results['intent_used']}")
    print(f"🤖 Model: {results['model_used']}")
    print(f"📚 Documents used: {results['retrieved_documents_count']}")
    print("-" * 80)
    
    print(f"\n💡 ZABITO's Response:")
    print("~" * 40)
    print(results['response'])
    print("~" * 40)
    
    if results['retrieved_documents_preview']:
        print(f"\n📋 Top Documents Referenced:")
        for doc in results['retrieved_documents_preview']:
            print(f"   #{doc['rank']} - {doc['file_name']} (Score: {doc['score']:.4f})")
            print(f"      Preview: {doc['text_preview']}")

def save_legal_response(results: Dict[str, Any], filename: str = "legal_response.json", mode: str = "append"):
    """Save legal assistant response to JSON file with append or overwrite mode"""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        directory_path = os.path.join(script_dir, "..", "responses")
        directory_path = os.path.normpath(directory_path)
        filepath = os.path.join(directory_path, filename)
        if mode == "append" and os.path.exists(filepath):
            # Load existing data
            with open(filepath, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            
            # Ensure existing data is a list
            if isinstance(existing_data, list):
                existing_data.append(results)
                data_to_save = existing_data
                print(f"📥 Appended to existing file with {len(existing_data)} responses")
            else:
                # If existing data is not a list, create a new list
                data_to_save = [existing_data, results]
                print("📥 Converted existing data to list and appended new response")
        else:
            # Create new list with current results
            data_to_save = [results]
            if mode == "append":
                print("📁 Creating new file for appending responses")
            else:
                print("📝 Overwrite mode: creating new file")
        
        # Save the data
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Legal response saved to: {filename} (mode: {mode})")
        print(f"📊 Total responses in file: {len(data_to_save)}")
        
    except json.JSONDecodeError:
        # If file exists but is invalid JSON, start fresh
        print("⚠️  File contains invalid JSON, creating new file...")
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump([results], f, indent=2, ensure_ascii=False)
        print(f"💾 Legal response saved to: {filename}")
        
    except Exception as e:
        print(f"❌ Error saving legal response: {e}")

# Also add this utility function to load previous responses
def load_legal_responses(filename: str = "legal_assistant_response.json") -> List[Dict[str, Any]]:
    """Load previous legal assistant responses from JSON file"""
    try:
        if not os.path.exists(filename):
            print(f"📭 No previous responses found at: {filename}")
            return []
        
        with open(filename, 'r', encoding='utf-8') as f:
            responses = json.load(f)
        
        if isinstance(responses, list):
            print(f"📖 Loaded {len(responses)} previous responses from: {filename}")
            return responses
        else:
            print(f"📖 Loaded 1 previous response from: {filename}")
            return [responses]
            
    except Exception as e:
        print(f"❌ Error loading legal responses: {e}")
        return []

# And add this function to display response history
def display_response_history(filename: str = "legal_assistant_response.json"):
    """Display history of legal assistant responses"""
    responses = load_legal_responses(filename)
    
    if not responses:
        print("📭 No response history found.")
        return
    
    print(f"\n📜 RESPONSE HISTORY ({len(responses)} queries)")
    print("=" * 80)
    
    for i, response in enumerate(responses, 1):
        print(f"\n#{i} - {response.get('query', 'Unknown query')}")
        print(f"   Intent: {response.get('intent_used', 'Unknown')}")
        print(f"   Documents: {response.get('retrieved_documents_count', 0)}")
        print(f"   Date: {response.get('timestamp', 'Unknown')}")
        print(f"   Preview: {response.get('response', '')[:100]}...")
        print("-" * 60)

# Integration with previous steps
def complete_legal_pipeline(query: str, use_gemini_embeddings: bool = True):
    """
    Complete pipeline: Query → Retrieval → Legal Assistant
    """
    print("🚀 COMPLETE LEGAL PIPELINE")
    print("=" * 60)
    
    # Step 1: Query Processing (choose based on embeddings)
    if use_gemini_embeddings:
        from gemini_queryProcessing import GeminiQueryProcessor, get_gemini_query_config
        query_config = get_gemini_query_config()
        query_processor = GeminiQueryProcessor(**query_config)
        query_result = query_processor.process_query(query)
    else:
        from queryProcessing import QueryProcessor
        query_processor = QueryProcessor()
        query_result = query_processor.process_query(query)
    
    # Step 2: Document Retrieval (choose based on embeddings)
    if use_gemini_embeddings:
        from gemini_chunk_retrieval import GeminiDocumentRetriever, get_gemini_retriever_config
        retrieval_config = get_gemini_retriever_config()
        retriever = GeminiDocumentRetriever(**retrieval_config)
        retrieval_results = retriever.process_query(query, top_k=5, use_pinecone=True)
    else:
        from chunk_retreival import DocumentRetriever, get_retriever_config
        retrieval_config = get_retriever_config()
        retriever = DocumentRetriever(**retrieval_config)
        retrieval_results = retriever.process_query(query, top_k=5, use_pinecone=True)
    
    # Step 3: Legal Assistant
    assistant_config = get_legal_assistant_config()
    legal_assistant = LegalAssistant(**assistant_config)
    
    final_results = legal_assistant.process_legal_query(
        query=query,
        retrieved_documents=retrieval_results['retrieved_documents'],
        intent_analysis=query_result.get('intent_analysis')
    )
    
    return final_results

# Test function
def test_legal_assistant():
    """Test the legal assistant with sample queries"""
    config = get_legal_assistant_config()
    assistant = LegalAssistant(**config)
    
    # Mock retrieved documents for testing
    mock_documents = [
        {
            'rank': 1,
            'score': 0.92,
            'text': 'Property registration in Sindh requires submission of sale deed, CNIC copies of buyer and seller, property documents, and payment of registration fees at the local registrar office. The process typically takes 7-10 working days.',
            'file_name': 'property_registration_guide.txt',
            'word_count': 45
        },
        {
            'rank': 2,
            'score': 0.87,
            'text': 'According to the Sindh Land Revenue Act, all property transfers must be registered with the relevant sub-registrar office. Unregistered properties may face legal challenges in court.',
            'file_name': 'sindh_land_act.txt',
            'word_count': 32
        }
    ]
    
    test_queries = [
        "What documents do I need for property registration in Sindh?",
        "How long does property registration take?",
        "Is property registration mandatory?"
    ]
    
    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"TESTING LEGAL ASSISTANT: '{query}'")
        print('='*80)
        
        results = assistant.process_legal_query(query, mock_documents)
        display_legal_response(results)
        
        # Save first test results
        if query == test_queries[0]:
            save_legal_response(results, "test_legal_response.json")

# Main interactive function
def main():
    """Main interactive legal assistant interface"""
    config = get_legal_assistant_config()
    assistant = LegalAssistant(**config)
    
    print("⚖️  ZABITO LEGAL ASSISTANT")
    print("=" * 50)
    print(f"Using model: {config['model_name']}")
    
    while True:
        print("\nOptions:")
        print("1. Use complete pipeline (Query → Retrieval → Response)")
        print("2. Test with mock documents")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '3':
            print("Goodbye!")
            break
            
        elif choice == '2':
            test_legal_assistant()
            continue
            
        elif choice == '1':
            query = input("Enter your legal question: ").strip()
            
            if not query:
                print("Please enter a question.")
                continue
                
            try:
                use_gemini = input("Use Gemini embeddings? (y/n, default y): ").strip().lower() != 'n'
                results = complete_legal_pipeline(query, use_gemini_embeddings=use_gemini)
                
                display_legal_response(results)
                
                # Ask to save results
                save_choice = input("\nSave response to file? (y/n): ").strip().lower()
                if save_choice == 'y':
                    filename = input("Filename (default: legal_response.json): ").strip() or "legal_response.json"
                    save_legal_response(results, filename)
                    
            except Exception as e:
                print(f"Error in legal pipeline: {e}")
                
        else:
            print("Invalid choice. Please enter 1-3.")

if __name__ == "__main__":
    # Install required packages
    # print("Required packages: pip install google-generativeai")
    
    # # Run test first
    # # print("🧪 Testing Legal Assistant...")
    # # test_legal_assistant()
    
    # # Then run interactive mode
    # print("\n\n🎮 Starting ZABITO Legal Assistant...")
    # main()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    directory_path = os.path.join(script_dir, "..", "responses")
    directory_path = os.path.normpath(directory_path)
    filename: str = "legal_response.json"
    print(os.path.join(directory_path, filename))
    print(os.path.exists(os.path.join(directory_path, filename)))