import os
import re
from typing import Dict, List

def extract_year_mapping_from_files(directory_path: str = ".") -> Dict[str, int]:
    """
    Extract year mapping from original 54 legal files
    
    Args:
        directory_path: Path where original files are located
        
    Returns:
        Dictionary mapping document names to years
        Example: {"SUCCESSION ACT 1925": 1925, "TRANSFER OF PROPERTY ACT 1882": 1882}
    """
    
    # Supported file extensions    
    year_mapping = {}
    
    print(f"📁 Scanning directory: {directory_path}")
    
    for filename in os.listdir(directory_path):
        # Skip chunk files and non-legal documents
            
        # Extract year using multiple patterns
        year_patterns = [
            r'(\d{4})',                    # Simple 4-digit year
            r'ACT\s+(\d{4})',              # ACT 1925
            r'Act\s+(\d{4})',              # Act 1925
            r'ACT+(\d{4})',                # ACT1925
            r'Act+(\d{4})',                # Act1925
            r'ACTS\s+(\d{4})',             # ACTS 1925
            r'Acts\s+(\d{4})',             # Acts 1925
            r'ACT,\s+(\d{4})',             # ACT, 1925
            r'Act,\s+(\d{4})',             # Act, 1925
            r'ACT,\s+(\d{4})+_',           # ACT, 1925_
            r'ACT.+(\d{4})',               # ACT.1925
            r'ACT_+(\d{4})',               # ACT_2023
            r'Act_+(\d{4})',               # Act_2023
            r'(\d{4})\s+ACT',              # 1925 ACT
            r'YEAR\s+(\d{4})',             # YEAR 1925
            r'ORDINANCE\s+(\d{4})',        # ORDINANCE 1925
            r'Ordinance\s+(\d{4})',        # Ordinance 1925
            r'ORDINANCE,\s+(\d{4})',       # ORDINANCE, 1925
            r'Ordinance,\s+(\d{4})',       # Ordinance, 1925
            r'Ordinance,+(\d{4})',         # Ordinance,1925
            r'ORDINANCE,+(\d{4})',         # ORDINANCE,1925
            r'ORDINANCE_+(\d{4})',         # ORDINANCE_2023
            r'Ordinance_+(\d{4})',         # Ordinance_1925
            r'ORDINANCE.+(\d{4})',         # ORDINANCE.2023
            r'Ordinance.+(\d{4})',         # Ordinance.2023
            r'LAW\s+(\d{4})',              # LAW 1925
            r'Law\s+(\d{4})',              # Law 1925
            r'LAWS\s+(\d{4})',             # LAWS 1925
            r'AMENDMENT\s+(\d{4})',        # AMENDMENT 2023
            r'Amendment\s+(\d{4})',        # Amendment 2023
            r'JURISDICTION_+(\d{4})',      # JURISDICTION_2023
            r'Jurisdiction_+(\d{4})',      # Jurisdiction_2023
            r'RULES-+(\d{4})+-',           # RULES-2023-
            r'rules-+(\d{4})+-',           # rules-2023-
            r'REGULATIONS,\s+(\d{4})',     # REGULATIONS, 2023
            r'Regulations,\s+(\d{4})',     # Regulations, 2023
            r'PROCEDURE\s+(\d{4})',        # PROCEDURE 2023
            r'Procedure\s+(\d{4})'         # Procedure 2023
        ]
        
        document_name = None
        extracted_year = None
        
        for pattern in year_patterns:
            match = re.search(pattern, filename)
            if match:
                year = int(match.group(1))
                # Validate it's a reasonable year
                if 1800 <= year <= 2024:
                    extracted_year = year
                    
                    # Extract clean document name
                    # Remove file extension
                    clean_name = os.path.splitext(filename)[0]
                    # Remove the year part for the document name
                    clean_name = re.sub(r'\s*\d{4}\s*', ' ', clean_name).strip()
                    document_name = clean_name
                    break
        
        if document_name and extracted_year:
            year_mapping[document_name] = extracted_year
            print(f"✅ {filename} → {document_name} ({extracted_year})")
        else:
            print(f"⚠️  Could not extract year from: {filename}")
    
    print(f"\n📊 Extracted {len(year_mapping)} document-year mappings")
    return year_mapping

# Additional utility function for manual verification
def verify_year_mapping(year_mapping: Dict[str, int]):
    """Verify and display the year mapping"""
    print("\n🔍 YEAR MAPPING VERIFICATION:")
    print("=" * 50)
    
    # Group by era for analysis
    eras = {
        "Pre-1900": [],
        "1900-1947 (British)": [],
        "1948-1999 (Early Pakistan)": [], 
        "2000-2019 (Modern)": [],
        "2020+ (Recent)": []
    }
    
    for doc_name, year in sorted(year_mapping.items(), key=lambda x: x[1]):
        if year < 1900:
            eras["Pre-1900"].append((doc_name, year))
        elif year < 1948:
            eras["1900-1947 (British)"].append((doc_name, year))
        elif year < 2000:
            eras["1948-1999 (Early Pakistan)"].append((doc_name, year))
        elif year < 2020:
            eras["2000-2019 (Modern)"].append((doc_name, year))
        else:
            eras["2020+ (Recent)"].append((doc_name, year))
    
    # Display by era
    for era, documents in eras.items():
        if documents:
            print(f"\n{era} ({len(documents)} documents):")
            for doc_name, year in documents:
                print(f"  {year}: {doc_name}")

# Usage example
if __name__ == "__main__":
    # Replace with your actual directory path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    directory_path = os.path.join(script_dir, "..", "data")
    directory_path = os.path.normpath(directory_path)
    
    mapping = extract_year_mapping_from_files(directory_path)
    verify_year_mapping(mapping)
    
    # Save to file for reference
    import json
    with open("document_year_mapping.json", "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"\n💾 Mapping saved to: document_year_mapping.json")