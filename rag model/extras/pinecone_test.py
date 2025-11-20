# pinecone_v3_test.py
from pinecone import Pinecone

def test_pinecone_v3():
    try:
        # Initialize Pinecone
        pc = Pinecone(api_key="your-pinecone-api-key", environment="your-environment")
        
        # List indexes
        indexes = pc.list_indexes()
        print("Available indexes:")
        for idx in indexes.indexes:
            print(f"  - {idx.name}")
        
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    test_pinecone_v3()