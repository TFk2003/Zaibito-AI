from code.controller.chunk import create_new_chunk
from code.controller.law_file import LawFileController
import os, re, nltk

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
    
    def add_law_file(self, pdf_path):
        controller = LawFileController()
        controller.add_law_file(pdf_path=pdf_path)

    def check_chunked_status(self, file):
        return file.chunked
    
    def mark_file_as_chunked(self, file_id):
        controller = LawFileController()
        updated_file = controller.update_file_chunked_status(file_id=file_id, chunked=True)
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
        
        # print(f"Cleaned text length: {len(text)} characters")
        return text
    
    def split_into_sentences(self, text):
        """
        Split text into sentences using NLTK with fallback
        """
        try:
            from nltk.tokenize import sent_tokenize
            sentences = sent_tokenize(text)
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
            from pdfminer.high_level import extract_text
            text = extract_text(pdf_path)
            return text
        except ImportError:
            print("pdfminer.six not available.")
            return ""
        
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
