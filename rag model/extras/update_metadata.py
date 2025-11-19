import os
import re
from pinecone import Pinecone
from dotenv import load_dotenv
from typing import Dict, Tuple
import json
from model_code.controller.chunk import get_all_embedded_chunks, get_chunk_details

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME")

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

def get_chunk_by_id(chunk_id):
    """Retrieve chunk details by chunk ID"""
    return get_chunk_details(chunk_id)

# ---------------------------------------------------------
# 1. Recreate EXACT Pinecone vector IDs (same as embedding)
# ---------------------------------------------------------
def recreate_vector_id(chunk):
    """
    Recreate original Pinecone vector ID exactly like the embedding script.
    format: <sanitized_file_name>_chunk_<chunk_id>
    """

    # THIS is the chunk_id used during embedding (comes from DB)
    chunk_id = str(chunk.id)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    directory_path = os.path.join(script_dir, "responses")
    directory_path = os.path.normpath(directory_path)
    file = os.path.join(directory_path, "document_year_mapping.json")
    # Load your document_year_mapping.json
    with open(file, "r", encoding="utf-8") as f:
        filename_year_map = json.load(f)

    # Law file name used earlier in embedding (comes from DB column)
    raw_name = chunk.chunk_name
    file_name_guess = re.sub(r'^chunk[_\-]*', '', raw_name, flags=re.IGNORECASE)
    file_name_guess = re.sub(r'_\d{1,4}\.txt$', '', file_name_guess)
    file_name_guess = file_name_guess.strip()
    law_name, law_year= parse_filename_for_law(file_name_guess, filename_year_map)
    safe_name = sanitize_filename(law_name)

    print(f"{safe_name}_chunk_{chunk_id}")
    return f"{safe_name}_chunk_{chunk_id}"
def sanitize_filename(name: str) -> str:
        """Make a short safe id-friendly file name."""
        s = re.sub(r'[^A-Za-z0-9\-_]+', '_', name).strip('_')
        return s[:120]  # keep it reasonably short

def parse_filename_for_law(filename: str, filename_year_map: Dict[str,int]=None) -> Tuple[str, int]:
        """
        Try to extract law_name, law_year and law_type from filename.
        - filename_year_map: optional mapping you provided to guarantee correct years.
        Returns (law_name, law_year (or None), law_type)
        """
        name = filename
        # If file stored with 'chunk_' prefix, remove it
        name = re.sub(r'^chunk[_\-]*', '', name, flags=re.IGNORECASE)
        # remove trailing chunk number if present (e.g. _015.txt)
        name = re.sub(r'_\d{1,4}\.txt$', '', name)
        name = name.replace('.txt', '').replace('.pdf', '')
        raw = name.strip()

        # try filename_year_map first
        if filename_year_map:
            for key in filename_year_map.keys():
                # case-insensitive match of key inside raw or raw inside key
                if key.lower() in raw.lower() or raw.lower() in key.lower():
                    return key, filename_year_map[key]

        # try to find a year pattern (4 digits between 1800-2050)
        year_match = re.search(r'(?<!\d)(18|19|20)\d{2}(?!\d)', raw)
        law_year = int(year_match.group(0)) if year_match else None

        # heuristics for law_name and law_type
        # law_name: drop year/time tokens and trailing stuff inside parentheses
        law_name = re.sub(r'\(\s*.*?\s*\)', '', raw)
        law_name = re.sub(r'[_\-]+', ' ', law_name)
        if law_year:
            law_name = re.sub(r'\b' + str(law_year) + r'\b', '', law_name)
        law_name = law_name.strip(' ,_-')

        if not law_name:
            law_name = raw

        return law_name, law_year

# ---------------------------------------------------------
# 2. Main Update Script
# ---------------------------------------------------------
def update_all_pinecone_chunks():
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY missing in .env")

    print("\n=== Connecting to Pinecone ===")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)

    print("Fetching embedded chunks from DB...")
    chunks = get_embedded_chunks()

    print(f"Total chunks found in DB: {len(chunks)}\n")

    updated = 0
    errors = 0
    missing_ids = []

    for chunk in chunks:
        try:
            vector_id = recreate_vector_id(chunk)

            full_text = chunk.chunk_data or ""

            # Pinecone update call
            index.update(
                id=vector_id,
                set_metadata={"full_text": full_text}
            )

            print(f"✔ Updated: {vector_id}")
            updated += 1

        except Exception as e:
            print(f"❌ Error updating chunk_id={chunk.id}: {e}")
            errors += 1
            missing_ids.append(vector_id)

    print("\n===========================")
    print(" Update Complete ")
    print("===========================")
    print(f"Total Updated: {updated}")
    print(f"Total Failed : {errors}")
    print("===========================")

    if missing_ids:
        with open("missing_update_ids.txt", "w", encoding="utf-8") as f:
            for mid in missing_ids:
                f.write(mid + "\n")

        print("Missing IDs saved to missing_update_ids.txt")

def update_single_chunk(chunk_id):
    """Update a single chunk in Pinecone by its chunk ID"""
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY missing in .env")
    print("\n=== Connecting to Pinecone ===")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)

    chunk = get_chunk_by_id(chunk_id)
    if not chunk:
        print(f"No chunk found with ID: {chunk_id}")
        return
    try:
        vector_id = f"Waqf_Properties_Ordinance_1979_chunk_{chunk.id}"

        full_text = chunk.chunk_data or ""

        # Pinecone update call
        index.update(
            id=vector_id,
            set_metadata={"full_text": full_text}
        )

        print(f"✔ Updated: {vector_id}")

    except Exception as e:
        print(f"❌ Error updating chunk_id={chunk.id}: {e}")

# ---------------------------------------------------------
# 3. Run the script
# ---------------------------------------------------------
if __name__ == "__main__":
    #update_all_pinecone_chunks()
    for cid in range(2613, 2626):
        update_single_chunk(cid)

