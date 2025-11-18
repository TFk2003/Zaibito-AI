import json
import os
from pinecone import Pinecone
from typing import Dict, List, Set

from model_code.controller.chunk import get_all_embedded_chunks

def query_pinecone_for_embedded_chunks(
    pinecone_api_key: str,
    index_name: str = "zabito-legal-gemini",
    embedded_mapping_file: str = "embedded_chunks_year_mapping.json"
) -> Dict[str, Dict]:
    """
    Query Pinecone to find which chunks are actually embedded and extract their metadata
    
    Args:
        pinecone_api_key: Pinecone API key
        index_name: Pinecone index name
        embedded_mapping_file: Path to embedded chunks mapping JSON file
        
    Returns:
        Dictionary with actual embedded chunks and their metadata
        Example: {
            "chunk_SUCCESSION ACT 1925_050.txt": {
                "vector_id": "chunk_123456789",
                "current_metadata": {...},
                "year": 1925
            }
        }
    """
    
    # Load embedded chunks mapping
    try:
        with open(embedded_mapping_file, 'r') as f:
            embedded_mapping = json.load(f)
        print(f"✅ Loaded {len(embedded_mapping)} expected embedded chunks")
    except FileNotFoundError:
        print(f"❌ File not found: {embedded_mapping_file}")
        return {}
    
    # Initialize Pinecone
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(index_name)
    
    print(f"🔍 Querying Pinecone index: {index_name}")
    
    # Get index stats to understand the data
    stats = index.describe_index_stats()
    total_vectors = stats['total_vector_count']
    print(f"📊 Index stats: {total_vectors} total vectors")
    
    actual_embedded_chunks = {}
    found_chunks = set()
    
    # Strategy: Query with dummy vectors to retrieve samples and match filenames
    # Since we can't query by filename directly, we'll use multiple approaches
    
    # Approach 1: Query with neutral vector to get random samples
    print("\n🔄 Approach 1: Sampling queries to find chunks...")
    
    # Create a neutral query vector (3072 dimensions for Gemini)
    neutral_vector = [0.02] * 3072
    
    # Query multiple times to get different samples
    for query_attempt in range(5):
        try:
            results = index.query(
                vector=neutral_vector,
                top_k=960,  # Get 960 vectors per query
                include_metadata=True
            )
            
            for match in results['matches']:
                filename = match['metadata'].get('file_name')
                if filename and filename in embedded_mapping:
                    if filename not in actual_embedded_chunks:
                        actual_embedded_chunks[filename] = {
                            'vector_id': match['id'],
                            'current_metadata': match['metadata'],
                            'year': embedded_mapping[filename]
                        }
                        found_chunks.add(filename)
                        print(f"✅ Found: {filename}")
            
        except Exception as e:
            print(f"❌ Query attempt {query_attempt + 1} failed: {e}")
    
    # Approach 2: Try to fetch by namespace if used
    print("\n🔄 Approach 2: Checking namespaces...")
    try:
        # If you used namespaces during embedding
        namespaces = stats.get('namespaces', {})
        for namespace in namespaces.keys():
            if namespace:  # Skip empty namespace
                try:
                    namespace_results = index.query(
                        vector=neutral_vector,
                        top_k=50,
                        include_metadata=True,
                        namespace=namespace
                    )
                    
                    for match in namespace_results['matches']:
                        filename = match['metadata'].get('file_name')
                        if filename and filename in embedded_mapping:
                            if filename not in actual_embedded_chunks:
                                actual_embedded_chunks[filename] = {
                                    'vector_id': match['id'],
                                    'current_metadata': match['metadata'],
                                    'year': embedded_mapping[filename]
                                }
                                found_chunks.add(filename)
                                print(f"✅ Found in namespace {namespace}: {filename}")
                                
                except Exception as e:
                    print(f"❌ Namespace {namespace} query failed: {e}")
                    
    except Exception as e:
        print(f"❌ Namespace check failed: {e}")
    
    # Analysis
    print(f"\n📊 RESULTS:")
    print(f"✅ Found {len(actual_embedded_chunks)} out of {len(embedded_mapping)} expected chunks")
    print(f"📈 Coverage: {(len(actual_embedded_chunks) / len(embedded_mapping)) * 100:.1f}%")
    
    if len(actual_embedded_chunks) < len(embedded_mapping):
        missing_count = len(embedded_mapping) - len(actual_embedded_chunks)
        print(f"⚠️  {missing_count} chunks not found in Pinecone")
        
        # Show first few missing chunks
        missing_chunks = set(embedded_mapping.keys()) - found_chunks
        print(f"Sample missing chunks: {list(missing_chunks)[:5]}")
    
    return actual_embedded_chunks

def save_actual_embedded_chunks(actual_chunks: Dict[str, Dict], filename: str = "actual_embedded_chunks.json"):
    """Append new embedded chunks to a JSON file without adding duplicates."""
    try:
        # Load existing data if the file exists
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                existing_data = json.load(f)
        else:
            existing_data = {}

        new_count = 0

        # Add only non-duplicate entries
        for chunk_id, data in actual_chunks.items():
            if chunk_id not in existing_data:
                existing_data[chunk_id] = data
                new_count += 1  # track how many were added

        # Save updated JSON
        with open(filename, 'w') as f:
            json.dump(existing_data, f, indent=2)

        print(f"💾 Added {new_count} new chunks (duplicates skipped) → {filename}")

    except Exception as e:
        print(f"❌ Error saving actual embedded chunks: {e}")

def get_embedded_chunks():
    """Retrieve all embedded chunks from the database"""
    embedded_chunks = get_all_embedded_chunks()
    return embedded_chunks

def find_missing_chunks(pinecone_api_key: str,
    index_name: str = "zabito-legal-gemini"):

    # Initialize Pinecone
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(index_name)

    

def analyze_actual_chunks(actual_chunks: Dict[str, Dict]):
    """Analyze the chunks found in Pinecone"""
    print("\n🔍 ANALYSIS OF FOUND CHUNKS:")
    
    # Year distribution
    year_distribution = {}
    for chunk_info in actual_chunks.values():
        year = chunk_info['year']
        year_distribution[year] = year_distribution.get(year, 0) + 1
    
    print("📊 Year distribution of found chunks:")
    for year, count in sorted(year_distribution.items()):
        print(f"   {year}: {count} chunks")
    
    # Metadata analysis
    metadata_fields = set()
    for chunk_info in actual_chunks.values():
        metadata_fields.update(chunk_info['current_metadata'].keys())
    
    print(f"📋 Metadata fields present: {list(metadata_fields)}")

# Usage example
if __name__ == "__main__":
    # Your configuration
    config = {
        'pinecone_api_key': "pcsk_UfHqv_QJNAVD1nz7ZxAXhdMS75q2Def87Ty9xcUh7qBXi6GrET4W8WHPEcWJvTMCiyJdG",
        'index_name': "zabito-legal-gemini",
        'embedded_mapping_file': "embedded_chunks_year_mapping.json"
    }
    
    # Step 3: Query Pinecone for actual embedded chunks
    actual_embedded_chunks = query_pinecone_for_embedded_chunks(**config)
    
    # Save results
    save_actual_embedded_chunks(actual_embedded_chunks)
    
    # Analyze results
    analyze_actual_chunks(actual_embedded_chunks)
    
    print(f"\n🎉 Step 3 completed!")
    print(f"   Found {len(actual_embedded_chunks)} chunks ready for metadata update")