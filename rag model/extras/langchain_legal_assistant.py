import os
import json
from typing import List, Dict, Any
from datetime import datetime
from langchain_pinecone import Pinecone
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import PromptTemplate
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains.conversational_retrieval.base import ConversationalRetrievalChain
import google.generativeai as genai

class GeminiEmbeddings(Embeddings):
    """Custom Gemini Embeddings for LangChain"""
    
    def __init__(self, google_api_key: str, model: str = "gemini-embedding-001"):
        self.google_api_key = google_api_key
        self.model = model
        genai.configure(api_key=google_api_key)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search documents."""
        embeddings = []
        for text in texts:
            response = genai.embed_content(
                model=self.model,
                content=text
            )
            embeddings.append(response['embedding'])
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        response = genai.embed_content(
            model=self.model,
            content=text
        )
        return response['embedding']

class LegalAssistantLangChain:
    def __init__(self, google_api_key: str, pinecone_api_key: str, 
                 index_name: str = "zabito-legal-gemini", use_gemini_embeddings: bool = False):
        self.google_api_key = google_api_key
        self.pinecone_api_key = pinecone_api_key
        self.index_name = index_name
        self.use_gemini_embeddings = use_gemini_embeddings
        self.vectorstore = None
        self.qa_chain = None
        self.memory = None
        self.retriever = None
        
    def initialize_components(self):
        """Initialize all LangChain components"""
        print("🔄 Initializing LangChain Legal Assistant...")
        
        # Initialize embeddings
        if self.use_gemini_embeddings:
            print("🔤 Using Gemini Embeddings...")
            embeddings = GeminiEmbeddings(
                google_api_key=self.google_api_key,
                model="gemini-embedding-001"
            )
        else:
            print("🔤 Using Local Sentence Transformers...")
            embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        
        # Initialize vector store
        print("🗂️ Connecting to Pinecone...")
        from pinecone import Pinecone as PineconeClient
        from langchain_pinecone import PineconeVectorStore
        pc = PineconeClient(api_key=self.pinecone_api_key)
        #print(f"Pinecone indexes available: {pc.list_indexes()}")
        # Get the index
        index = pc.Index(self.index_name)
        
        # Create vectorstore from existing index
        self.vectorstore = PineconeVectorStore(
            index=index,
            embedding=embeddings,
            text_key="text_preview"  # The metadata field where text is stored
        )
        
        # Initialize LLM
        print("🧠 Initializing Gemini LLM...")
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=self.google_api_key,
            temperature=0.1,
            convert_system_message_to_human=True
        )
        
        # Initialize memory for conversation history
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key='answer'
        )
        
        # Create custom legal prompt
        legal_prompt = self._create_legal_prompt_template()
        
        # Initialize retriever with advanced configuration
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",  # Max Marginal Relevance for diversity
            search_kwargs={
                "k": 5,  # Number of documents to retrieve
                "fetch_k": 10,  # Number of documents to fetch before MMR
                "lambda_mult": 0.7,  # Diversity parameter
            }
        )
        
        # Create QA chain
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self.retriever,
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": legal_prompt},
            return_source_documents=True,
            verbose=True
        )
        
        print("✅ LangChain Legal Assistant initialized successfully!")
    
    def _create_legal_prompt_template(self) -> PromptTemplate:
        """Create specialized legal prompt template"""
        template = """You are ZABITO, an expert legal assistant specialized in Sindh property laws and Pakistani legal procedures.

CONTEXT INFORMATION:
{context}

CONVERSATION HISTORY:
{chat_history}

USER QUESTION: {question}

CRITICAL LEGAL GUIDELINES:
1. Answer STRICTLY based on the provided legal context above
2. If information is not in the context, clearly state "Based on the provided documents, this information is not available"
3. Do not use external knowledge or make assumptions about Pakistani law
4. Be precise, factual, and cite relevant document sections when possible
5. Highlight if certain procedures might have been updated or amended recently
6. Structure your response clearly with bullet points for complex procedures
7. Include important caveats about legal compliance and documentation

RESPONSE FORMAT:
- Start with a clear, direct answer to the question
- Reference specific legal provisions or document sections when applicable
- Use bullet points for procedures, requirements, or steps
- Include warnings about common legal pitfalls if relevant
- End with a summary of key legal considerations

ZABITO Legal Response:"""
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "question", "chat_history"]
        )
    
    def query_legal_assistant(self, question: str) -> Dict[str, Any]:
        """Query the legal assistant with a question"""
        if not self.qa_chain:
            self.initialize_components()
        
        print(f"🔍 Processing legal query: '{question}'")
        
        try:
            # Execute the query
            result = self.qa_chain.invoke({"question": question})
            
            # Extract source documents information
            source_docs = []
            for i, doc in enumerate(result.get("source_documents", [])):
                source_docs.append({
                    "rank": i + 1,
                    "source": doc.metadata.get("file_name", "Unknown"),
                    "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "score": getattr(doc, 'score', 'N/A')
                })
            
            # Prepare comprehensive response
            response_data = {
                "question": question,
                "answer": result["answer"],
                "source_documents": source_docs,
                "documents_retrieved": len(source_docs),
                "timestamp": datetime.now().isoformat(),
                "embedding_model": "Gemini" if self.use_gemini_embeddings else "SentenceTransformer",
                "retrieval_method": "MMR (Max Marginal Relevance)"
            }
            
            print(f"✅ Response generated with {len(source_docs)} source documents")
            return response_data
            
        except Exception as e:
            error_msg = f"❌ Error in legal query: {str(e)}"
            print(error_msg)
            return {
                "question": question,
                "answer": f"I apologize, but I encountered an error while processing your legal query. Please try again.",
                "error": str(e),
                "source_documents": [],
                "documents_retrieved": 0,
                "timestamp": datetime.now().isoformat()
            }
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get the conversation history"""
        if not self.memory:
            return []
        
        history = []
        for i, message in enumerate(self.memory.chat_history):
            role = "user" if i % 2 == 0 else "assistant"
            history.append({
                "role": role,
                "content": message.content,
                "timestamp": datetime.now().isoformat()
            })
        
        return history
    
    def clear_conversation_history(self):
        """Clear the conversation history"""
        if self.memory:
            self.memory.clear()
            print("🗑️ Conversation history cleared")

class AdvancedLegalRetriever:
    """Advanced retriever with multiple search strategies"""
    
    def __init__(self, vectorstore: Pinecone):
        self.vectorstore = vectorstore
        
    def semantic_search(self, query: str, k: int = 5) -> List[Document]:
        """Standard semantic search"""
        return self.vectorstore.similarity_search(query, k=k)
    
    def mmr_search(self, query: str, k: int = 5, fetch_k: int = 10) -> List[Document]:
        """Diverse search using Max Marginal Relevance"""
        return self.vectorstore.max_marginal_relevance_search(
            query, k=k, fetch_k=fetch_k
        )
    
    def search_with_filters(self, query: str, metadata_filter: Dict, k: int = 5) -> List[Document]:
        """Search with metadata filters"""
        return self.vectorstore.similarity_search(
            query, k=k, filter=metadata_filter
        )

# Configuration
def get_langchain_config():
    """Get LangChain configuration"""
    return {
        'google_api_key': "",
        'pinecone_api_key': "",
        'index_name': "your-index-name",  # or "zabito-legal-gemini" for local
        'use_gemini_embeddings': True
    }

# Utility functions
def display_langchain_response(response: Dict[str, Any]):
    """Display LangChain response in formatted way"""
    print(f"\n🎯 LANGCHAIN LEGAL RESPONSE")
    print("=" * 80)
    print(f"📝 Question: {response['question']}")
    print(f"🤖 Model: Gemini-2.5-flash")
    print(f"🔤 Embeddings: {response.get('embedding_model', 'Unknown')}")
    print(f"🔍 Retrieval: {response.get('retrieval_method', 'Unknown')}")
    print(f"📚 Documents: {response['documents_retrieved']}")
    print(f"⏰ Time: {response['timestamp']}")
    print("-" * 80)
    
    print(f"\n💡 ZABITO's Answer:")
    print("~" * 40)
    print(response['answer'])
    print("~" * 40)
    
    if response['source_documents']:
        print(f"\n📋 Source Documents:")
        for doc in response['source_documents']:
            print(f"   #{doc['rank']} - {doc['source']}")
            print(f"      Preview: {doc['content_preview']}")

def save_langchain_response(response: Dict[str, Any], filename: str = "langchain_legal_responses.json"):
    """Save LangChain response with append mode"""
    try:
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            if isinstance(existing_data, list):
                existing_data.append(response)
                data_to_save = existing_data
            else:
                data_to_save = [existing_data, response]
        else:
            data_to_save = [response]
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, indent=2, ensure_ascii=False)
        
        print(f"💾 LangChain response saved to: {filename}")
        
    except Exception as e:
        print(f"❌ Error saving LangChain response: {e}")

# Test function
def test_langchain_assistant():
    """Test the LangChain legal assistant"""
    config = get_langchain_config()
    assistant = LegalAssistantLangChain(**config)
    
    test_questions = [
        "What documents are required for property registration in Sindh?",
        "Explain the lease agreement process for commercial properties",
        "What are the legal requirements for property transfer?",
        "How long does property registration typically take in Karachi?"
    ]
    
    for question in test_questions:
        print(f"\n{'='*80}")
        print(f"TESTING LANGCHAIN: '{question}'")
        print('='*80)
        
        response = assistant.query_legal_assistant(question)
        display_langchain_response(response)
        
        # Save first response
        if question == test_questions[0]:
            save_langchain_response(response)

# Main interactive function
def main():
    """Main interactive LangChain legal assistant"""
    config = get_langchain_config()
    assistant = LegalAssistantLangChain(**config)
    
    print("⚖️  ZABITO LEGAL ASSISTANT (LangChain)")
    print("=" * 60)
    print(f"Using: {config['index_name']}")
    print(f"Embeddings: {'Gemini' if config['use_gemini_embeddings'] else 'Local'}")
    print(f"Retrieval: MMR with diversity")
    
    while True:
        print("\nOptions:")
        print("1. Ask legal question")
        print("2. View conversation history")
        print("3. Clear conversation history")
        #print("4. Test with sample questions")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '5':
            print("Goodbye!")
            break
            
        elif choice == '2':
            history = assistant.get_conversation_history()
            if history:
                print(f"\n📜 Conversation History ({len(history)//2} exchanges):")
                for msg in history[-6:]:  # Show last 3 exchanges
                    print(f"   {msg['role'].upper()}: {msg['content'][:100]}...")
            else:
                print("📭 No conversation history")
                
        elif choice == '3':
            assistant.clear_conversation_history()
            print("✅ Conversation history cleared")
            
        # elif choice == '4':
        #     test_langchain_assistant()
            
        elif choice == '1':
            question = input("\nEnter your legal question: ").strip()
            
            if not question:
                print("Please enter a question.")
                continue
                
            try:
                response = assistant.query_legal_assistant(question)
                display_langchain_response(response)
                
                # Ask to save
                save_choice = input("\nSave response? (y/n): ").strip().lower()
                if save_choice == 'y':
                    filename = input("Filename (default: langchain_responses.json): ").strip() or "langchain_responses.json"
                    save_langchain_response(response, filename)
                    
            except Exception as e:
                print(f"❌ Error: {e}")
                
        else:
            print("Invalid choice. Please enter 1-5.")

if __name__ == "__main__":
    # Install required packages
    print("Required packages:")
    print("pip install langchain langchain-community langchain-google-genai pinecone-client")
    
    # Run test
    # print("\n🧪 Testing LangChain Legal Assistant...")
    # test_langchain_assistant()
    
    # Run interactive
    print("\n\n🎮 Starting LangChain Interactive Mode...")
    main()