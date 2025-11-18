import json
import re
from typing import Dict, List, Tuple

def reconstruct_chunk_to_year_mapping(document_year_mapping_file: str = "document_year_mapping.json") -> Dict[str, int]:
    """
    Reconstruct chunk-to-year mapping using the document-year mapping
    
    Args:
        document_year_mapping_file: Path to the document-year mapping JSON file
        
    Returns:
        Dictionary mapping chunk filenames to years
        Example: {"chunk_SUCCESSION ACT 1925_050.txt": 1925}
    """
    
    # Load document-year mapping
    try:
        with open(document_year_mapping_file, 'r') as f:
            document_year_mapping = json.load(f)
        print(f"✅ Loaded {len(document_year_mapping)} document-year mappings")
    except FileNotFoundError:
        print(f"❌ File not found: {document_year_mapping_file}")
        return {}
    except json.JSONDecodeError:
        print(f"❌ Invalid JSON in: {document_year_mapping_file}")
        return {}
    
    # Reconstruct chunk filenames and map to years
    chunk_year_mapping = {}
    unmatched_documents = set()
    
    print("🔍 Reconstructing chunk-to-year mapping...")
    
    for document_name, year in document_year_mapping.items():
        # Generate possible chunk filename patterns for this document
        # Remove any extra spaces and normalize
        clean_doc_name = re.sub(r'\s+', ' ', document_name).strip()
        
        # Create chunk filename pattern: chunk_{document_name}_{number}.txt
        # We'll create multiple patterns to handle different numbering
        
        # Pattern 1: Direct match with the document name
        chunk_pattern = f"chunk_{clean_doc_name}_"
        
        # For each document, assume multiple chunks (001 to 999)
        # We'll create entries for common chunk ranges
        for chunk_num in range(1, 100):  # Adjust range based on your needs
            chunk_filename = f"{chunk_pattern}{chunk_num:03d}.txt"
            chunk_year_mapping[chunk_filename] = year
        
        print(f"📄 {clean_doc_name} → {year} (chunks: 001-{chunk_num:03d})")
    
    print(f"✅ Generated {len(chunk_year_mapping)} chunk-year mappings")
    
    # Verify we have coverage for expected chunks
    expected_chunk_count = 2625  # Your total chunks
    coverage_percentage = (len(chunk_year_mapping) / expected_chunk_count) * 100
    print(f"📊 Coverage: {len(chunk_year_mapping)}/{expected_chunk_count} chunks ({coverage_percentage:.1f}%)")
    
    return chunk_year_mapping

def get_embedded_chunks_year_mapping(chunk_year_mapping: Dict[str, int], 
                                   embedded_chunks_count: int = 960) -> Dict[str, int]:
    """
    Extract mapping for only the embedded chunks (first 960)
    
    Args:
        chunk_year_mapping: Full chunk-to-year mapping
        embedded_chunks_count: Number of chunks that are embedded
        
    Returns:
        Dictionary mapping only embedded chunk filenames to years
    """
    
    # Sort chunks to get consistent ordering (by document, then chunk number)
    sorted_chunks = sorted(chunk_year_mapping.keys())
    
    # Take first N chunks as the embedded ones
    embedded_mapping = {}
    for chunk_filename in sorted_chunks[:embedded_chunks_count]:
        embedded_mapping[chunk_filename] = chunk_year_mapping[chunk_filename]
    
    print(f"✅ Extracted {len(embedded_mapping)} embedded chunk mappings")
    
    # Analyze year distribution of embedded chunks
    year_distribution = {}
    for chunk_filename, year in embedded_mapping.items():
        year_distribution[year] = year_distribution.get(year, 0) + 1
    
    print("\n📊 Embedded Chunks Year Distribution:")
    for year, count in sorted(year_distribution.items()):
        print(f"   {year}: {count} chunks")
    
    return embedded_mapping

# Utility function to save the mappings
def save_chunk_mappings(chunk_year_mapping: Dict[str, int], 
                       embedded_mapping: Dict[str, int]):
    """Save both full and embedded chunk mappings to files"""
    
    # Save full mapping
    with open("full_chunk_year_mapping.json", "w") as f:
        json.dump(chunk_year_mapping, f, indent=2)
    print(f"💾 Full chunk mapping saved: full_chunk_year_mapping.json")
    
    # Save embedded chunks mapping
    with open("embedded_chunks_year_mapping.json", "w") as f:
        json.dump(embedded_mapping, f, indent=2)
    print(f"💾 Embedded chunks mapping saved: embedded_chunks_year_mapping.json")

# Usage example
if __name__ == "__main__":
    # Step 2: Reconstruct chunk-to-year mapping
    chunk_year_mapping = reconstruct_chunk_to_year_mapping("document_year_mapping.json")
    
    # Extract only embedded chunks mapping
    embedded_mapping = get_embedded_chunks_year_mapping(chunk_year_mapping, embedded_chunks_count=960)
    
    # Save the mappings
    save_chunk_mappings(chunk_year_mapping, embedded_mapping)
    
    print(f"\n🎉 Step 2 completed!")
    print(f"   Total chunks mapped: {len(chunk_year_mapping)}")
    print(f"   Embedded chunks mapped: {len(embedded_mapping)}")