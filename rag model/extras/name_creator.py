import re, os
import json
from typing import Dict, Tuple

def recreate_vector_id(chunk):
    """
    Recreate original Pinecone vector ID exactly like the embedding script.
    format: <sanitized_file_name>_chunk_<chunk_id>
    """

    # THIS is the chunk_id used during embedding (comes from DB)
    chunk_id = str(chunk.id)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    directory_path = os.path.join(script_dir,"..", "responses")
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

if __name__ == "__main__":
    recreate_vector_id(type('Chunk', (object,), {'id': 2614, 'chunk_name': 'chunk_Waqf Properties Ordinance 1979_002.txt'})())