import re
from pinecone import Pinecone

from model_code.controller.chunk import get_all_embedded_chunks

def chunk_name_to_pinecone_id(chunk_id, chunk_name):
    """
    Convert database chunk_name to Pinecone ID format
    
    Example:
    chunk_Agha Khan Properties(Succession and Transfer) ACT, 2025_001.txt
    -> Agha_Khan_Properties_Succession_and_Transfer_ACT_2025_chunk_1
    """
    # Remove 'chunk_' prefix and '.txt' suffix
    name = chunk_name.replace('chunk_', '').replace('.txt', '')
    
    # Extract the chunk number (001 -> 1)
    match = re.search(r'_(\d+)$', name)
    if match:
        chunk_num = str(int(match.group(1)))  # Convert 001 to 1
        name = name[:match.start()]  # Remove _001 from end
    else:
        chunk_num = "1"
    
    # Replace special characters and spaces with underscores
    # Remove parentheses and commas
    name = name.replace('(', '_').replace(')', '').replace(',', '').replace('&','_').replace('.', '_')
    name = name.replace(' ', '_')
    
    # Remove multiple consecutive underscores
    name = re.sub(r'_+', '_', name)
    
    # Construct final ID
    pinecone_id = f"{name}_chunk_{chunk_id}"
    print(f"Converted chunk_name '{chunk_name}' to Pinecone ID '{pinecone_id}'")
    return pinecone_id


def find_missing_chunks(pc_index, db_embedded_chunks):
    """
    Find chunks that exist in DB but not in Pinecone
    
    Args:
        pc_index: Pinecone index object
        db_embedded_chunks: List of chunk dicts from database
    
    Returns:
        List of missing chunk IDs and their details
    """
    print(f"Total chunks in DB marked as embedded: {len(db_embedded_chunks)}")
    
    # Convert all DB chunk names to Pinecone IDs
    db_to_pinecone_map = {}
    pinecone_ids = []
    
    for chunk in db_embedded_chunks:
        chunk_name = chunk.chunk_name
        pinecone_id = chunk_name_to_pinecone_id(chunk_name)
        db_to_pinecone_map[pinecone_id] = chunk
        pinecone_ids.append(pinecone_id)
    
    print(f"Converted {len(pinecone_ids)} chunk names to Pinecone IDs")
    
    # Check Pinecone in batches (max 1000 per fetch)
    missing_chunks = []
    batch_size = 100
    
    for i in range(0, len(pinecone_ids), batch_size):
        batch = pinecone_ids[i:i+batch_size]
        print(f"Checking batch {i//batch_size + 1} ({len(batch)} IDs)...")
        
        try:
            # Fetch from Pinecone
            fetch_response = pc_index.fetch(ids=batch)
            # Access vectors using dot notation or to_dict()
            fetched_vectors = fetch_response.vectors if hasattr(fetch_response, 'vectors') else fetch_response.to_dict().get('vectors', {})
            fetched_ids = set(fetched_vectors.keys())

            print(f"  Found {len(fetched_ids)}/{len(batch)} vectors in Pinecone")
            
            # Find missing IDs
            missing_in_batch = set(batch) - fetched_ids

            if missing_in_batch:
                print(f"  ⚠️ Missing {len(missing_in_batch)} vectors in this batch")
            
            for missing_id in missing_in_batch:
                original_chunk = db_to_pinecone_map[missing_id]
                missing_chunks.append({
                    'db_id': original_chunk.id,
                    'chunk_name': original_chunk.chunk_name,
                    'pinecone_id': missing_id,
                    'file_id': original_chunk.file_id
                })
                print(f"  ❌ Missing: {missing_id}")
        
        except Exception as e:
            print(f"Error fetching batch: {e}")
            continue
    
    return missing_chunks


def update_db_for_missing_chunks(cursor, connection, missing_chunks):
    """
    Update database to mark missing chunks as not embedded
    
    Args:
        cursor: Database cursor
        connection: Database connection
        missing_chunks: List of missing chunk details
    """
    if not missing_chunks:
        print("✅ No missing chunks found! DB and Pinecone are in sync.")
        return
    
    print(f"\n🔍 Found {len(missing_chunks)} missing chunks:")
    for chunk in missing_chunks:
        print(f"  - DB ID: {chunk['db_id']}, Name: {chunk['chunk_name']}")
    
    # Update database
    confirm = input(f"\nUpdate {len(missing_chunks)} chunks to embedded=FALSE? (yes/no): ")
    
    if confirm.lower() == 'yes':
        for chunk in missing_chunks:
            cursor.execute(
                "UPDATE chunks SET embedded = FALSE WHERE id = ?",
                (chunk['db_id'],)
            )
        
        connection.commit()
        print(f"✅ Updated {len(missing_chunks)} chunks in database")
        
        # Verify counts
        cursor.execute("SELECT COUNT(*) FROM chunks WHERE embedded = TRUE")
        db_count = cursor.fetchone()[0]
        print(f"\n📊 New DB count: {db_count} embedded chunks")
    else:
        print("❌ Update cancelled")

def get_embedded_chunks():
    """Retrieve all embedded chunks from the database"""
    all_chunks = []
    skip = 0
    limit = 100

    while True:
        chunk_page = get_all_embedded_chunks(skip=skip, limit=limit)
        if not chunk_page:
            break
        
        all_chunks.extend(chunk_page)
        skip += limit
    return all_chunks

# Main execution
def main():
    # Initialize Pinecone
    pc = Pinecone(api_key="pcsk_UfHqv_QJNAVD1nz7ZxAXhdMS75q2Def87Ty9xcUh7qBXi6GrET4W8WHPEcWJvTMCiyJdG")
    index = pc.Index("zabito-legal-gemini")
    
    # Pinecone list() output → list of lists → flatten
    raw_ids = index.list()
    pinecone_ids_actual = set(id for sub in raw_ids for id in sub)
    print("Pinecone vectors:", len(pinecone_ids_actual))

    # DB chunks
    embedded_chunks = get_embedded_chunks()

    db_ids_expected = {
        chunk_name_to_pinecone_id(c.id,c.chunk_name)
        for c in embedded_chunks
    }
    print("DB embedded chunks:", len(db_ids_expected))

    # Compare
    missing = db_ids_expected - pinecone_ids_actual
    with open("missing_ids.txt", "w", encoding="utf-8") as f:
        for m in missing:
            f.write(m + "\n")

    print("\nTotal missing:", len(missing))

    # Find missing chunks
    #missing_chunks = find_missing_chunks(index, embedded_chunks)
    
    #print(missing_chunks)


if __name__ == "__main__":
    main()