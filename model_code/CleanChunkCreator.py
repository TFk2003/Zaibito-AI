import os, re, nltk
from unittest import result
import pytesseract
from pdf2image import convert_from_path
import pdfplumber

class TextProcessor:
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
    
    def _setup_nltk(self):
        """Setup NLTK data with proper error handling"""
        try:
            nltk.data.find('tokenizers/punkt_tab')
            print("NLTK punkt_tab found successfully")
        except LookupError:
            print("Downloading NLTK punkt_tab...")
            try:
                nltk.download('punkt_tab', quiet=True)
                print("NLTK punkt_tab downloaded successfully")
            except Exception as e:
                print(f"Error downloading NLTK data: {e}")
                print("Falling back to simple sentence splitting...")
    
    def clean_text(self, text):
        """
        Clean extracted text by removing extra spaces, normalizing line breaks, etc.
        """
        if not text:
            return ""
        
        print(f"Original text length: {len(text)} characters")
        
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
            print("Using simple sentence splitting...")
            sentences = self._simple_sentence_split(text)
        
        return [sentence.strip() for sentence in sentences if sentence.strip()]
    
    def _simple_sentence_split(self, text):
        """Simple sentence splitting without NLTK"""
        # Split on periods, question marks, and exclamation marks followed by space and capital
        sentences = re.split(r'([.!?])\s+(?=[A-Z])', text)
        
        # Reconstruct sentences with their punctuation
        reconstructed = []
        i = 0
        while i < len(sentences):
            if i + 1 < len(sentences) and sentences[i+1] in ['.', '!', '?']:
                reconstructed.append(sentences[i] + sentences[i+1])
                i += 2
            else:
                if sentences[i].strip():
                    reconstructed.append(sentences[i])
                i += 1
        
        # Further split long sentences
        final_sentences = []
        for sentence in reconstructed:
            # If sentence is too long, split on commas
            if len(sentence.split()) > 50:
                parts = re.split(r',\s+', sentence)
                # Recombine parts that are too short
                temp_parts = []
                current_part = ""
                for part in parts:
                    if current_part:
                        test_combined = current_part + ", " + part
                    else:
                        test_combined = part
                    
                    if len(test_combined.split()) <= 50:
                        current_part = test_combined
                    else:
                        if current_part:
                            temp_parts.append(current_part)
                        current_part = part
                
                if current_part:
                    temp_parts.append(current_part)
                
                final_sentences.extend([p.strip() for p in temp_parts if p.strip()])
            else:
                final_sentences.append(sentence.strip())
        
        print(f"Simple splitting created {len(final_sentences)} sentences")
        return final_sentences
    
    def calculate_word_count(self, text):
        """Calculate word count for text"""
        words = text.split()
        return len(words)
    
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
            #print(f"{current_word_count=}, {sentence_word_count=}, {self.chunk_size=}, {self.chunk_overlap=}")
            # If adding this sentence doesn't exceed chunk size
            if current_word_count + sentence_word_count <= self.chunk_size:
                current_chunk.append(sentence)
                current_word_count += sentence_word_count
                i += 1
                #print(f"Added sentence to chunk: '{sentence[:30]}...' (Total words: {current_word_count})")
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
                    #print(f"Created chunk with {current_word_count} words and {len(current_chunk)} sentences")
                
                # Handle overlap for next chunk
                if self.chunk_overlap > 0 and current_chunk:
                    # Take some sentences from the end of current chunk for overlap
                    overlap_sentences = []
                    overlap_word_count = 0
                    
                    #print(f"Creating overlap with up to {self.chunk_overlap} words")
                    for sent in reversed(current_chunk):
                        sent_word_count = self.calculate_word_count(sent)
                        #print(f"Evaluating sentence for overlap: '{sent[:30]}...' (Words: {sent_word_count})")
                        if overlap_word_count + sent_word_count <= self.chunk_overlap:
                            overlap_sentences.insert(0, sent)
                            overlap_word_count += sent_word_count
                            #print(f"Added sentence to overlap: '{sent[:30]}...' (Total overlap words: {overlap_word_count})")
                        else:
                            break
                    
                    current_chunk = overlap_sentences
                    current_word_count = overlap_word_count
                    #print(f"Overlap created with {overlap_word_count} words and {len(overlap_sentences)} sentences")
                else:
                    current_chunk = []
                    current_word_count = 0
                    #print("Starting a new chunk")
        
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
    
    def process_text(self, text):
        """Complete text processing pipeline"""
        print("Cleaning text...")
        cleaned_text = self.clean_text(text)
        
        print("Splitting into sentences...")
        sentences = self.split_into_sentences(cleaned_text)
        
        print("Creating chunks...")
        chunks = self.create_chunks(sentences)
        
        return cleaned_text, chunks

# Alternative: Simple processor without NLTK
class SimpleTextProcessor:
    def __init__(self, chunk_size=400, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def clean_text(self, text):
        """Clean text without NLTK"""
        if not text:
            return ""
        
        print(f"Original text length: {len(text)} characters")
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\-\:\(\)\;\"]', '', text)
        
        # Fix broken sentences
        text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)
        
        text = text.strip()
        print(f"Cleaned text length: {len(text)} characters")
        return text
    
    def split_sentences(self, text):
        """Simple sentence splitting without NLTK"""
        # Improved sentence splitting
        sentences = []
        current_sentence = ""
        
        for char in text:
            current_sentence += char
            if char in '.!?':
                # Check if this is likely the end of a sentence
                if len(current_sentence) > 10:  # Minimum sentence length
                    sentences.append(current_sentence.strip())
                    current_sentence = ""
        
        # Add the last sentence if any
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
        
        print(f"Split into {len(sentences)} sentences")
        return sentences
    
    def word_count(self, text):
        """Simple word count"""
        return len(text.split())
    
    def create_chunks(self, text):
        """Create chunks from cleaned text"""
        sentences = self.split_sentences(text)
        chunks = []
        current_chunk = []
        current_word_count = 0
        
        for sentence in sentences:
            sentence_word_count = self.word_count(sentence)
            
            if current_word_count + sentence_word_count <= self.chunk_size:
                current_chunk.append(sentence)
                current_word_count += sentence_word_count
            else:
                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    chunks.append({
                        'text': chunk_text,
                        'word_count': current_word_count,
                        'char_count': len(chunk_text)
                    })
                
                # Start new chunk
                current_chunk = [sentence]
                current_word_count = sentence_word_count
        
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'word_count': current_word_count,
                'char_count': len(chunk_text)
            })
        
        return chunks

# PDF extraction function
def extract_text_from_pdf(pdf_path):
    """Extract text from PDF using pdfminer"""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            print(f"Extracting text from {pdf_path}...")
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        
        if text.strip():
            print(f"Extracted Direct text length: {len(text)} characters")
            return text
    except ImportError:
        print("pdfminer.six not available. Using fallback PDF extraction...")
        # Fallback PDF extraction code would go here
        return ""
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

# Main processing function
def process_pdf_to_chunks(pdf_path, chunk_size=400, chunk_overlap=50, use_nltk=True):
    """
    Complete pipeline: PDF → Text → Clean → Chunks
    """
    # Extract text from PDF
    raw_text = extract_text_from_pdf(pdf_path)
    
    if not raw_text:
        print("No text extracted from PDF")
        return "", []
    
    # Choose processor
    if use_nltk:
        processor = TextProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    else:
        processor = SimpleTextProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        # For SimpleTextProcessor, process differently
        cleaned_text = processor.clean_text(raw_text)
        chunks = processor.create_chunks(cleaned_text)
        return cleaned_text, chunks
    
    # Process the text
    cleaned_text, chunks = processor.process_text(raw_text)
    
    return cleaned_text, chunks

# Utility functions
def analyze_chunks(chunks):
    """Analyze chunk statistics"""
    if not chunks:
        print("No chunks to analyze")
        return
    
    total_words = sum(chunk['word_count'] for chunk in chunks)
    avg_words = total_words / len(chunks)
    
    print(f"\n=== CHUNK ANALYSIS ===")
    print(f"Total chunks: {len(chunks)}")
    print(f"Total words: {total_words}")
    print(f"Average words per chunk: {avg_words:.1f}")
    print(f"Chunk size range: {min(chunk['word_count'] for chunk in chunks)} - {max(chunk['word_count'] for chunk in chunks)} words")
    
    # Show first few chunks as preview
    print(f"\n=== CHUNK PREVIEWS ===")
    for i, chunk in enumerate(chunks[:3], 1):  # Show first 3 chunks
        print(f"Chunk {i} ({chunk['word_count']} words):")
        print(f"  {chunk['text'][:100]}...")
        print()

def save_chunks(chunks, base_filename="chunk"):
    """Save chunks to individual files"""
    for i, chunk in enumerate(chunks, 1):
        filename = f"{base_filename}_{i:03d}.txt"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(chunk['text'])
        print(f"Saved: {filename} ({chunk['word_count']} words)")

# Example usage
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    directory_path = os.path.join(script_dir, "..", "data")
    directory_path = os.path.normpath(directory_path)
    pdf_file = "Stamp Act, 1899.pdf"  # Your PDF file
    pdf_file = os.path.join(directory_path, pdf_file)
    try:
        print("Starting PDF processing...")
        
        # Try with NLTK first, fallback to simple if needed
        try:
            cleaned_text, chunks = process_pdf_to_chunks(pdf_file, chunk_size=400, chunk_overlap=30, use_nltk=True)
        except Exception as e:
            print(f"NLTK processing failed, trying simple method: {e}")
            cleaned_text, chunks = process_pdf_to_chunks(pdf_file, chunk_size=300, chunk_overlap=30, use_nltk=False)
        
        if chunks:
            analyze_chunks(chunks)
            file_name = os.path.basename(pdf_file)
            save_chunks(chunks, base_filename="chunk_"+file_name.removesuffix(".pdf"))
            print(f"\nSuccessfully processed {len(chunks)} chunks!")
        else:
            print("No chunks were created")
        
    except Exception as e:
        print(f"Error processing PDF: {e}")