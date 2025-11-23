from pdf2image import convert_from_path
import pdfplumber, pytesseract
from model_code.controller.chunk import create_new_chunk
from model_code.controller.law_file import LawFileController
import os, re, nltk, json

from model_code.models.law_file import LawFile

class ChunkCreator:
    def __init__(self, chunk_size=400, chunk_overlap=50):
        """
        Initialize TextProcessor
        
        Args:
            chunk_size (int): Target number of words per chunk
            chunk_overlap (int): Number of overlapping words between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._setup_nltk()
    
    def stored_files(self):
        controller = LawFileController()
        files = controller.list_law_files()
        return files
    
    def get_file_id_by_name(self, file_name):
        controller = LawFileController()
        return controller.get_file_id_by_name(file_name=file_name)
    
    def get_file_by_id(self, file_id):
        controller = LawFileController()
        return controller.get_file_by_id(file_id=file_id)

    def add_law_file(self, pdf_path):
        controller = LawFileController()
        controller.add_law_file(pdf_path=pdf_path)
        print(f"Added law file: {pdf_path}")

    def check_chunked_status(self, file: LawFile):
        return file.chunked
    
    def mark_file_as_chunked(self, file_id):
        controller = LawFileController()
        updated_file = controller.update_law_file_chunked_status(file_id=file_id, chunked=True)
        print(f"Marked file ID {file_id} as chunked.")
        return updated_file
    
    def get_directory_files(self, directory_path):
        files = []
        for filename in os.listdir(directory_path):
            if filename.endswith(".pdf"):
                files.append(os.path.join(directory_path, filename))
        return files
    
    def _setup_nltk(self):
        """Setup NLTK data with proper error handling"""
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            print("Downloading NLTK punkt_tab...")
            try:
                nltk.download('punkt_tab', quiet=True)
            except Exception as e:
                print(f"Error downloading NLTK data: {e}")
                print("Falling back to simple sentence splitting...")

    def clean_text(self, text):
        """
        Clean extracted text by removing extra spaces, normalizing line breaks, etc.
        """
        if not text:
            return ""
        
        # print(f"Original text length: {len(text)} characters")
        
        # Remove excessive whitespace (multiple spaces, tabs, newlines)
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\-\:\(\)\;\"]', '', text)
        
        # Fix line breaks in the middle of sentences
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)  # Fix hyphenated words
        text = re.sub(r'(\w)\s*\n\s*(\w)', r'\1 \2', text)    # Fix broken sentences
        
        # Normalize multiple periods/ellipses
        text = re.sub(r'\.{2,}', '...', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        print(f"Cleaned text length: {len(text)} characters")
        return text
    
    def split_sentence(self, sentence, max_words=300):
        result = []
        queue = [sentence]

        while queue:
            s = queue.pop(0)
            if len(s.split()) > max_words:
                mid = len(s) // 2
                queue.append(s[:mid])
                queue.append(s[mid:])
            else:
                result.append(s)
        return result

    def split_into_sentences(self, text):
        """
        Split text into sentences using NLTK with fallback
        """
        try:
            from nltk.tokenize import sent_tokenize
            sentences = sent_tokenize(text)
            sen = []
            for sentence in sentences:
                sen.extend(self.split_sentence(sentence))
            sentences = sen
            print(f"Successfully split into {len(sentences)} sentences using NLTK")
        except Exception as e:
            print(f"NLTK sentence tokenization failed: {e}")
            # print("Using simple sentence splitting...")
            # sentences = self._simple_sentence_split(text)
        
        return [sentence.strip() for sentence in sentences if sentence.strip()]

    def process_text(self, text):
        """Complete text processing pipeline"""
        cleaned_text = self.clean_text(text)
        sentences = self.split_into_sentences(cleaned_text)
        chunks = self.create_chunks(sentences)
        
        return chunks
    
    def create_chunks(self, sentences):
        """Create chunks from sentences with overlapping"""
        if not sentences:
            return []
        
        chunks = []
        current_chunk = []
        current_word_count = 0
        
        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            sentence_word_count = self.calculate_word_count(sentence)
            
            # If adding this sentence doesn't exceed chunk size
            if current_word_count + sentence_word_count <= self.chunk_size:
                current_chunk.append(sentence)
                current_word_count += sentence_word_count
                i += 1
            else:
                # If current chunk is not empty, save it
                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    chunks.append({
                        'text': chunk_text,
                        'word_count': current_word_count,
                        'sentence_count': len(current_chunk),
                        'char_count': len(chunk_text)
                    })
                
                # Handle overlap for next chunk
                if self.chunk_overlap > 0 and current_chunk:
                    # Take some sentences from the end of current chunk for overlap
                    overlap_sentences = []
                    overlap_word_count = 0
                    
                    for sent in reversed(current_chunk):
                        sent_word_count = self.calculate_word_count(sent)
                        if overlap_word_count + sent_word_count <= self.chunk_overlap:
                            overlap_sentences.insert(0, sent)
                            overlap_word_count += sent_word_count
                        else:
                            break
                    
                    current_chunk = overlap_sentences
                    current_word_count = overlap_word_count
                else:
                    current_chunk = []
                    current_word_count = 0
        
        # Add the last chunk if it exists
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'word_count': current_word_count,
                'sentence_count': len(current_chunk),
                'char_count': len(chunk_text)
            })
        
        print(f"Created {len(chunks)} chunks from text")
        return chunks
    
    def calculate_word_count(self, text):
        """Calculate word count for text"""
        words = text.split()
        return len(words)
    
    def process_pdf_to_chunks(self,pdf_path, chunk_size=400, chunk_overlap=50, use_nltk=True):
        """
        Complete pipeline: PDF → Text → Clean → Chunks
        """
        # Extract text from PDF
        raw_text = self.extract_text_from_pdf(pdf_path)
        
        if not raw_text:
            print("No text extracted from PDF")
            return []
        
        # Choose processor
        if use_nltk:
            processor = ChunkCreator(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            chunks = processor.process_text(raw_text)
        
        return chunks

    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF using pdfminer"""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                       text += page_text + "\n"
        
            if text.strip():
                print(f"Extracted text from: {pdf_path}")
                return text
        except:
            text = ""
            try:
                images = convert_from_path(pdf_path, dpi=200)
                # OCR each page
                for i, image in enumerate(images):
                    page_text = pytesseract.image_to_string(image, lang='eng')
                    text += f"Page {i+1}:\n{page_text}\n"
                print(f"OCR extracted text length: {len(text)} characters")
            except Exception as e:
                print(f"OCR processing failed: {e}")
        return text        
        
    def save_chunks(self, chunks, file_id, base_filename="chunk"):
        """Save chunks to individual files"""
        for i, chunk in enumerate(chunks, 1):
            filename = f"{base_filename}_{i:03d}.txt"
            create_new_chunk(
                chunk_name=filename,
                chunk_data=chunk['text'],
                file_id=file_id 
            )
            print(f"Saved: {filename} ({chunk['word_count']} words)")

    def save_year_mapping_to_file(self, filename):
        """Save year mapping to a JSON file"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        directory_path = os.path.join(script_dir, "responses")
        directory_path = os.path.normpath(directory_path)
        file = os.path.join(directory_path, "document_year_mapping.json")
        # Load your document_year_mapping.json
        with open(file, "r", encoding="utf-8") as f:
            filename_year_map = json.load(f)
        
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
                if 1800 <= year <= 2026:
                    extracted_year = year
                    
                    # Extract clean document name
                    # Remove file extension
                    clean_name = os.path.splitext(filename)[0]
                    # Remove the year part for the document name
                    document_name = clean_name
                    break
        
        if document_name and extracted_year:
            filename_year_map[document_name] = extracted_year
            with open(file, "w", encoding="utf-8") as f:
                json.dump(filename_year_map, f, indent=2)
            print(f"✅ {filename} → {document_name} ({extracted_year})")
    