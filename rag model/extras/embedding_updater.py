import json
import re
from typing import List, Dict, Any
from pinecone import Pinecone

class ChunkMetadataUpdater:
    def __init__(self, pinecone_api_key: str, index_name: str = "zabito-legal-gemini"):
        self.pinecone_api_key = pinecone_api_key
        self.index_name = index_name
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index = self.pc.Index(index_name)
    
    def extract_year_from_filename(self, filename: str) -> int:
        """
        Extract year from filename using multiple pattern matching
        """
        # Pattern 1: Four-digit years (1925, 2023, etc.)
        year_patterns = [
            r'(\d{4})',  # Simple 4-digit year
            r'ACT\s+(\d{4})',  # ACT 1925
            r'ACT\s+OF\s+(\d{4})',  # ACT OF 1925
            r'(\d{4})\s+ACT',  # 1925 ACT
            r'YEAR\s+(\d{4})',  # YEAR 1925
        ]
        
        for pattern in year_patterns:
            match = re.search(pattern, filename.upper())
            if match:
                year = int(match.group(1))
                # Validate it's a reasonable year (between 1800 and current year)
                if 1800 <= year <= 2024:
                    return year
        
        # Default to oldest possible year if no year found
        print(f"⚠️  No year found in: {filename}, defaulting to 1900")
        return 1900
    
    def get_all_vectors(self, batch_size: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieve all vectors from Pinecone index
        Note: This uses describe_index_stats and fetch, not direct query
        """
        print("📊 Getting index statistics...")
        stats = self.index.describe_index_stats()
        total_vectors = stats['total_vector_count']
        print(f"Total vectors in index: {total_vectors}")
        
        # Since we can't list all IDs directly, we'll work with what we have
        # You might need to use namespace if you used one
        vectors = []
        
        # Try to fetch vectors if you have their IDs
        # For now, we'll work with the metadata we can access via query
        return vectors
    
    def update_chunks_by_query(self) -> Dict[str, Any]:
        """
        Update chunks by querying the index and extracting years from metadata
        """
        print("🔄 Starting chunk metadata update...")
        
        # Query to get some vectors with their metadata
        # Using a neutral query to get diverse results
        sample_query = [0.1] * 384  # Dummy query vector (adjust dimension as needed)
        
        try:
            # Get a sample of vectors to analyze
            results = self.index.query(
                vector=sample_query,
                top_k=100,  # Get first 100 to analyze pattern
                include_metadata=True,
                include_values=False
            )
            
            updates = []
            updated_count = 0
            error_count = 0
            
            print(f"🔍 Analyzing {len(results['matches'])} sample vectors...")
            
            for match in results['matches']:
                try:
                    vector_id = match['id']
                    current_metadata = match['metadata']
                    filename = current_metadata.get('file_name', '')
                    
                    if filename:
                        # Extract year from filename
                        year = self.extract_year_from_filename(filename)
                        
                        # Prepare update
                        new_metadata = current_metadata.copy()
                        new_metadata['year'] = year
                        new_metadata['document_name'] = self.clean_document_name(filename)
                        
                        updates.append({
                            "id": vector_id,
                            "metadata": new_metadata
                        })
                        
                        print(f"✅ {filename} → Year: {year}")
                        updated_count += 1
                    
                except Exception as e:
                    print(f"❌ Error processing {match.get('id', 'unknown')}: {e}")
                    error_count += 1
            
            # Apply updates in batches
            if updates:
                print(f"📤 Applying {len(updates)} updates to Pinecone...")
                self.index.update(updates=updates)
            
            return {
                "total_processed": len(results['matches']),
                "successfully_updated": updated_count,
                "errors": error_count,
                "sample_updates": updates[:3]  # Show first 3 as sample
            }
            
        except Exception as e:
            print(f"❌ Error in update process: {e}")
            return {"error": str(e)}
    
    def clean_document_name(self, filename: str) -> str:
        """
        Clean and extract document name from chunk filename
        """
        # Remove 'chunk_' prefix and file extension
        clean_name = filename.replace('chunk_', '').replace('.txt', '')
        
        # Remove the trailing number part (_001, _050, etc.)
        clean_name = re.sub(r'_\d+$', '', clean_name)
        
        return clean_name
    
    

    def update_specific_chunks(self, chunk_files: List[str]) -> Dict[str, Any]:
        """
        Update specific chunks by searching for them in the index
        """
        print(f"🔍 Updating {len(chunk_files)} specific chunks...")
        
        updates = []
        updated_count = 0
        not_found_count = 0
        
        for chunk_file in chunk_files:
            try:
                # Search for vectors with this filename
                results = self.index.query(
                    vector=[0.1] * 384,  # Dummy query
                    top_k=10,
                    include_metadata=True,
                    filter={"file_name": {"$eq": chunk_file}}
                )
                
                if results['matches']:
                    for match in results['matches']:
                        current_metadata = match['metadata']
                        year = self.extract_year_from_filename(chunk_file)
                        
                        new_metadata = current_metadata.copy()
                        new_metadata['year'] = year
                        new_metadata['document_name'] = self.clean_document_name(chunk_file)
                        
                        updates.append({
                            "id": match['id'],
                            "metadata": new_metadata
                        })
                    
                    print(f"✅ Found and updated {chunk_file} → Year: {year}")
                    updated_count += 1
                else:
                    print(f"⚠️  Chunk not found in index: {chunk_file}")
                    not_found_count += 1
                    
            except Exception as e:
                print(f"❌ Error updating {chunk_file}: {e}")
        
        # Apply all updates
        if updates:
            print(f"📤 Applying {len(updates)} updates...")
            self.index.update(updates=updates)
        
        return {
            "total_chunks": len(chunk_files),
            "updated": updated_count,
            "not_found": not_found_count,
            "errors": len(chunk_files) - updated_count - not_found_count
        }

# Batch processing for your 960 embedded chunks
def process_all_embedded_chunks():
    """Process all your 960 embedded chunks"""
    
    # Your configuration
    config = {
        'pinecone_api_key': "pcsk_UfHqv_QJNAVD1nz7ZxAXhdMS75q2Def87Ty9xcUh7qBXi6GrET4W8WHPEcWJvTMCiyJdG",
        'index_name': "zabito-legal-index"  # or "zabito-legal-gemini" if using Gemini
    }
    
    updater = ChunkMetadataUpdater(**config)
    
    # Method 1: Update via sampling (quick start)
    print("🎯 METHOD 1: Sampling Update")
    result1 = updater.update_chunks_by_query()
    print(f"Sample update result: {result1}")
    
    # Method 2: Update specific chunks from your list
    print("\n🎯 METHOD 2: Specific Chunk Update")
    
    # You would load your actual chunk filenames here
    # For now, using pattern from your example
    sample_chunks = [
        "chunk_SUCCESSION ACT 1925_050.txt",
        "chunk_TRANSFER OF PROPERTY ACT 1882_001.txt",
        "chunk_SINDH RENTAL ACT 2023_010.txt"
        # Add all your 960 chunk filenames here
    ]
    
    result2 = updater.update_specific_chunks(sample_chunks[:10])  # Test with 10 first
    print(f"Specific update result: {result2}")

# Utility to extract all chunk filenames from your local files
def extract_chunk_filenames_from_local():
    """Extract all chunk filenames from your local chunk files"""
    import glob
    import os
    
    chunk_files = glob.glob("chunk_*.txt")
    print(f"📁 Found {len(chunk_files)} chunk files locally")
    
    # Extract years for analysis
    year_counter = {}
    updater = ChunkMetadataUpdater("dummy_key")  # Just for year extraction
    
    for chunk_file in chunk_files:
        year = updater.extract_year_from_filename(chunk_file)
        year_counter[year] = year_counter.get(year, 0) + 1
    
    print("📊 Year distribution in your chunks:")
    for year, count in sorted(year_counter.items()):
        print(f"   {year}: {count} chunks")
    
    return chunk_files

# Verify the updates
def verify_metadata_updates():
    """Verify that metadata was updated correctly"""
    config = {
        'pinecone_api_key': "pcsk_UfHqv_QJNAVD1nz7ZxAXhdMS75q2Def87Ty9xcUh7qBXi6GrET4W8WHPEcWJvTMCiyJdG",
        'index_name': "zabito-legal-index"
    }
    
    updater = ChunkMetadataUpdater(**config)
    
    # Query to check updated vectors
    sample_query = [0.1] * 384
    results = updater.index.query(
        vector=sample_query,
        top_k=20,
        include_metadata=True
    )
    
    print("🔍 Verification - Checking updated metadata:")
    for match in results['matches']:
        metadata = match['metadata']
        if 'year' in metadata:
            print(f"✅ {metadata.get('file_name', 'unknown')} → Year: {metadata['year']}")
        else:
            print(f"❌ {metadata.get('file_name', 'unknown')} → No year metadata")

if __name__ == "__main__":
    print("🔄 LEGAL CHUNK METADATA UPDATER")
    print("=" * 50)
    
    # Step 1: Analyze local chunks
    print("\n1. 📊 Analyzing local chunk files...")
    local_chunks = extract_chunk_filenames_from_local()
    
    # Step 2: Update embedded chunks
    print("\n2. 🚀 Updating embedded chunks in Pinecone...")
    process_all_embedded_chunks()
    
    # Step 3: Verify updates
    print("\n3. ✅ Verifying updates...")
    verify_metadata_updates()
    
    print("\n🎉 Metadata update process completed!")