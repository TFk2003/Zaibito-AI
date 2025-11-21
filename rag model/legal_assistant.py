import os
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv
from langchain_pinecone import Pinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import PromptTemplate
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains.conversational_retrieval.base import ConversationalRetrievalChain

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME       = os.getenv("INDEX_NAME")
GOOGLE_API_KEY   = os.getenv("GOOGLE_API_KEY")
EMBED_MODEL      = os.getenv("EMBEDDING_MODEL")
GEN_AI_MODEL     = os.getenv("GEN_AI_MODEL")

class GeminiEmbeddings(Embeddings):
    """Custom Gemini Embeddings for LangChain"""
    
    def __init__(self, google_api_key: str, model: str):
        self.embeddings_model = GoogleGenerativeAIEmbeddings(
            model=model,
            google_api_key=google_api_key # Pass the API key if not in env vars
        )
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search documents."""
        embeddings = self.embeddings_model.embed_documents(texts)
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        embedding = self.embeddings_model.embed_query(text)
        return embedding

class YearAwareRetriever(BaseRetriever):
    """
    Custom retriever that prioritizes recent laws while maintaining semantic relevance.
    Uses a hybrid scoring approach: semantic_score + year_recency_boost
    """
    
    vectorstore: Any
    k: int = 5
    fetch_k: int = 20
    year_weight: float = 0.3  # Weight for year recency (0-1)
    use_mmr: bool = True
    lambda_mult: float = 0.7
    current_year: int = datetime.now().year
    metadata_filter: Optional[Dict[str, Any]] = None  # NEW: Add metadata filtering
    
    def _calculate_year_score(self, law_year: Optional[int]) -> float:
        """
        Calculate recency score based on law year.
        More recent laws get higher scores.
        """
        if not law_year or law_year == 0:
            return 0.0
        
        try:
            law_year = int(float(law_year))
        except (ValueError, TypeError):
            return 0.0
        
        if law_year == 0 or law_year > self.current_year + 1:
            return 0.0
        # Calculate years difference
        years_old = self.current_year - law_year
        
        # Exponential decay: newer laws get higher scores
        # Laws from this year get 1.0, laws from 50+ years ago get ~0.0
        decay_rate = 0.05  # Adjust this to change how quickly old laws lose priority
        year_score = max(0.0, 1.0 - (years_old * decay_rate))
        
        return year_score
    
    def _rerank_by_year(self, documents: List[tuple]) -> List[Document]:
        """
        Rerank documents based on a combination of:
        1. Original similarity score
        2. Law year recency

        Args:
            documents: List of tuples (Document, similarity_score)
        """
        scored_docs = []
        
        for doc in documents:
            # Get original similarity score (if available)
            if isinstance(doc, tuple):
                doc, similarity_score = doc
            else:
                doc = doc
                # Try to get score from metadata or default
                similarity_score = 0.5
            
            # Get law year from metadata
            law_year = doc.metadata.get('law_year')
            if isinstance(law_year, str):
                try:
                    law_year = int(law_year)
                except (ValueError, TypeError):
                    law_year = None
            
            # Calculate year recency score
            year_score = self._calculate_year_score(law_year)
            
            # Hybrid score: weighted combination
            # semantic_weight + year_weight should = 1.0
            semantic_weight = 1.0 - self.year_weight
            final_score = (semantic_weight * similarity_score) + (self.year_weight * year_score)
            
            # Store scores in metadata for debugging
            doc.metadata['similarity_score'] = similarity_score
            doc.metadata['year_score'] = year_score
            doc.metadata['final_score'] = final_score
            
            scored_docs.append((doc, final_score))
        
        # Sort by final score (descending)
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k documents
        return [doc for doc, score in scored_docs[:self.k]]
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Get documents relevant to the query, prioritizing recent laws."""
        
        # Always use similarity_search_with_score to get actual scores
        try:
            # Get documents with similarity scores and optional metadata filter
            search_kwargs = {"k": self.fetch_k}
            if self.metadata_filter:
                search_kwargs["filter"] = self.metadata_filter
                
            docs_with_scores = self.vectorstore.similarity_search_with_score(
                query, 
                **search_kwargs
            )
            
            # Attach scores to document objects
            processed_docs = []
                # Pinecone returns distance, convert to similarity (higher is better)
                # For cosine similarity: similarity = 1 - distance
            for doc, distance in docs_with_scores:
                similarity_score = max(0.0, min(1.0, 1.0 - distance))
                processed_docs.append((doc, similarity_score))
                
        except Exception as e:
            print(f"⚠️ Warning: Could not get similarity scores: {e}")
            print(f"⚠️ Falling back to standard search without scores")
            # Fallback to MMR without scores
            docs = self.vectorstore.similarity_search(
                query, 
                k=self.fetch_k
            )
            processed_docs = [(doc, 0.5) for doc in docs]
        
        # Rerank by year
        reranked_docs = self._rerank_by_year(processed_docs)
        
        return reranked_docs


class LegalAssistantLangChain:
    def __init__(self, google_api_key: str, pinecone_api_key: str, 
                 index_name: str , use_gemini_embeddings: bool = False,
                 year_weight: float = 0.3, prioritize_recent: bool = True):
        self.google_api_key = google_api_key
        self.pinecone_api_key = pinecone_api_key
        self.index_name = index_name
        self.use_gemini_embeddings = use_gemini_embeddings
        self.year_weight = year_weight
        self.prioritize_recent = prioritize_recent
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
                model=EMBED_MODEL
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
            text_key="full_text"  # The metadata field where text is stored
        )
        
        # Initialize LLM
        print("🧠 Initializing Gemini LLM...")
        llm = ChatGoogleGenerativeAI(
            model=GEN_AI_MODEL,
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
        if self.prioritize_recent:
            print(f"📅 Using Year-Aware Retrieval (year_weight={self.year_weight})...")
            # Optional: Add metadata filter for property-related documents only
            # metadata_filter = {"is_property_related": True}  # Uncomment to filter
            metadata_filter = None  # Set to None for no filtering

            self.retriever = YearAwareRetriever(
                vectorstore=self.vectorstore,
                k=5,
                fetch_k=30,
                year_weight=self.year_weight,
                use_mmr=True,
                lambda_mult=0.8,
                metadata_filter=metadata_filter
            )
        else:
            print("🔍 Using Standard MMR Retrieval...")
            self.retriever = self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 5,
                    "fetch_k": 10,
                    "lambda_mult": 0.7,
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
        """Create specialized legal prompt template with year awareness"""
        template = """You are ZABITO, an expert legal assistant specialized in Sindh property laws and Pakistani legal procedures.

CONTEXT INFORMATION (Documents ranked by relevance and recency - MOST RECENT LAWS FIRST):
{context}

CONVERSATION HISTORY:
{chat_history}

USER QUESTION: {question}

CRITICAL LEGAL GUIDELINES:
1. **ALWAYS PRIORITIZE THE MOST RECENT LAW**: When answering, lead with the most recent applicable law (highest year)
2. **CLEARLY STATE THE LAW AND YEAR**: Format as "According to [Law Name] ([Year])..."
3. **FLAG SUPERSEDED LAWS**: If older laws are in the context, mention if they've been replaced/amended
4. **ONLY USE THE CONTEXT ABOVE**: Answer EXCLUSIVELY based on the documents provided above. Do NOT use external knowledge.
5. **BE EXPLICIT ABOUT GAPS**: If information is missing, state "Based on the provided documents, this information is not available"
6. **NO EXTERNAL KNOWLEDGE**: Do not use assumptions or knowledge beyond the provided documents
7. **IF NOT IN CONTEXT, SAY SO**: If the information is not in the provided documents, you MUST state: "This information is not available in the provided legal documents."
8. **NO HALLUCINATION**: Do not invent laws, sections, or regulations not in the context
9. **CITE SECTIONS WITH YEARS**: Always include the year when referencing laws (e.g., "Section 5, Aga Khan Properties Act 2025")
10. **STRUCTURE CLEARLY**: Use bullet points for procedures, requirements, or steps
11. **LEGAL WARNINGS**: Include caveats about compliance and documentation requirements

RESPONSE FORMAT:
✓ **Check Context First**: Before answering, verify the information is in the provided documents
✓ **Direct Answer**: Start with the most current legal position from the context
✓ **Law Citation**: Always format as "[Law Name] ([Year]), Section [X]"
✓ **Supersession Notice**: If context includes older versions, explicitly state: "Note: This replaces the previous [Old Law] ([Old Year])"
✓ **Step-by-Step**: For procedures, use numbered steps or clear bullet points
✓ **Warnings**: Highlight common legal pitfalls or compliance requirements
✓ **Summary**: End with key takeaways emphasizing the current legal position

EXAMPLE FORMAT:
"According to the [Most Recent Law Name] (2025), Section X, the procedure is:

1. [First step with specific requirements]
2. [Second step with documentation needed]
3. [Third step with timeline]

⚠️ Important: [Key warnings or caveats]

📌 Summary: As of 2025, the current legal requirement is [brief summary]."

EXAMPLE OF CORRECT RESPONSE:
"According to the [Law Name] (2025), Section 5 from the provided documents:
1. [Specific procedure from context]
2. [Specific requirement from context]

Note: The provided documents do not contain information about [missing aspect]."

EXAMPLE OF INCORRECT RESPONSE (DO NOT DO THIS):
"According to the Building Regulation Amendment (2019)..." [when this is NOT in the context]

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
                law_year = doc.metadata.get("law_year", "Unknown")
                source_docs.append({
                    "rank": i + 1,
                    "law_name": doc.metadata.get("law_name", "Unknown"),
                    "law_year": law_year,
                    "source": doc.metadata.get("file_name", "Unknown"),
                    "section": doc.metadata.get("section_number", "N/A"),
                    "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "similarity_score": doc.metadata.get("similarity_score", "N/A"),
                    "year_score": doc.metadata.get("year_score", "N/A"),
                    "final_score": doc.metadata.get("final_score", "N/A")
                })
            
            # Prepare comprehensive response
            response_data = {
                "question": question,
                "answer": result["answer"],
                "source_documents": source_docs,
                "documents_retrieved": len(source_docs),
                "timestamp": datetime.now().isoformat(),
                "embedding_model": "Gemini",
                "retrieval_method": f"Year-Aware MMR (weight={self.year_weight})" if self.prioritize_recent else "Standard MMR",
                "year_aware": self.prioritize_recent
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
        'google_api_key': GOOGLE_API_KEY or "",
        'pinecone_api_key': PINECONE_API_KEY or "",
        'index_name': INDEX_NAME or "your-index-name",  # or "zabito-legal-gemini" for local
        'use_gemini_embeddings': True,
        'year_weight': 0.3,  # Adjust: 0.0 = no year priority, 1.0 = only year matters
        'prioritize_recent': True
    }

# Utility functions
def display_langchain_response(response: Dict[str, Any]):
    """Display LangChain response in formatted way"""
    print(f"\n🎯 LANGCHAIN LEGAL RESPONSE")
    print("=" * 80)
    print(f"📝 Question: {response['question']}")
    print(f"🤖 Model: {GEN_AI_MODEL}")
    print(f"🔤 Embeddings: {response.get('embedding_model', 'Unknown')}")
    print(f"🔍 Retrieval: {response.get('retrieval_method', 'Unknown')}")
    print(f"📚 Documents: {response['documents_retrieved']}")
    print(f"📅 Year-Aware: {response.get('year_aware', False)}")
    print(f"⏰ Time: {response['timestamp']}")
    print("-" * 80)
    
    print(f"\n💡 ZABITO's Answer:")
    print("~" * 40)
    print(response['answer'])
    print("~" * 40)
    
    if response['source_documents']:
        print(f"\n📋 Source Documents:")
        for doc in response['source_documents']:
            print(f"\n   #{doc['rank']} - {doc['law_name']} ({doc['law_year']})")
            print(f"      File: {doc['source']}")
            print(f"      Section: {doc['section']}")
            if isinstance(doc['final_score'], float):
                print(f"      Scores - Semantic: {doc['similarity_score']:.3f} | Year: {doc['year_score']:.3f} | Final: {doc['final_score']:.3f}")
            print(f"      Preview: {doc['content_preview']}")


def save_langchain_response(response: Dict[str, Any], filename: str = "langchain_legal_responses.json"):
    """Save LangChain response with append mode"""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        directory_path = os.path.join(script_dir, "responses")
        directory_path = os.path.normpath(directory_path)
        filename = os.path.join(directory_path, filename)
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