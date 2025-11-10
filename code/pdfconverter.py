from pdfminer.high_level import extract_text

def pdf_to_text_simple(pdf_path, output_txt_path=None):
    """
    Simple PDF to text conversion using extract_text function
    """
    # Extract text from PDF
    text = extract_text(pdf_path)
    
    # Save to file if output path provided
    if output_txt_path:
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write(text)
    
    return text

# Usage
text = pdf_to_text_simple('PUB-NEW-23-000039.pdf', 'P23Text.txt')
print(text)