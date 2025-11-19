import os
import time, re, json, math, uuid
from model_code.core.config import settings
from model_code.controller.chunk import get_all_chunks, mark_chunk_embedded
from typing import List, Dict, Tuple, Any
from datetime import datetime
from collections import Counter

from model_code.models.chunk import Chunk

class EmbeddingCreator:
    def __init__(self):
        self.google_api_key = settings.GOOGLE_API_KEY
        self.embedding_model = settings.EMBEDDING_MODEL
        self.pinecone_api_key = settings.PINECONE_API_KEY
        self.index_name = settings.INDEX_NAME
        self.initialize_gemini()
        self.initialize_pinecone(dimension=3072)
    
    STOPWORDS = {
        # small handy stopword set; expand if needed
        "the","and","is","in","to","of","a","for","on","that","by","with","as","be",
        "this","which","or","are","an","at","from","may","it","such","shall","any",
        "have","has","were","was","not","but","their","they","its","these","there",
        "into","also","other","under","per","within"
    }

    def sanitize_filename(self, name: str) -> str:
        """Make a short safe id-friendly file name."""
        s = re.sub(r'[^A-Za-z0-9\-_]+', '_', name).strip('_')
        return s[:120]  # keep it reasonably short

    def parse_filename_for_law(self, filename: str, filename_year_map: Dict[str,int]=None) -> Tuple[str, int, str]:
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
                    return key, filename_year_map[key], self.infer_law_type_from_name(key)

        # try to find a year pattern (4 digits between 1800-2050)
        year_match = re.search(r'(?<!\d)(18|19|20)\d{2}(?!\d)', raw)
        law_year = int(year_match.group(0)) if year_match else None

        # heuristics for law_name and law_type
        law_type = self.infer_law_type_from_name(raw)
        # law_name: drop year/time tokens and trailing stuff inside parentheses
        law_name = re.sub(r'\(\s*.*?\s*\)', '', raw)
        law_name = re.sub(r'[_\-]+', ' ', law_name)
        if law_year:
            law_name = re.sub(r'\b' + str(law_year) + r'\b', '', law_name)
        law_name = law_name.strip(' ,_-')

        if not law_name:
            law_name = raw

        return law_name, law_year, law_type

    def infer_law_type_from_name(self, name: str) -> str:
        """Infer law type from filename text."""
        name_low = name.lower()
        for t in ["act", "ordinance", "rules", "regulations", "by-laws", "bylaws", "code", "schedule", "amendment", "ordinance","policy"]:
            if t in name_low:
                # normalize
                if "bylaw" in t or "by-laws" in t or "bylaws" in name_low:
                    return "By-Laws"
                if "regulation" in t:
                    return "Regulations"
                if "rule" in t:
                    return "Rules"
                if "act" in t:
                    return "Act"
                if "ordinance" in t:
                    return "Ordinance"
                if "code" in t:
                    return "Code"
                if "schedule" in t:
                    return "Schedule"
                if "amendment" in t:
                    return "Amendment"
                if "policy" in t:
                    return "Policy"
        # default
        return "Unknown"

    def extract_section_number(self, text: str) -> str:
        """
        Find the first likely section reference in text.
        Looks for 'Section 5', 's. 5', 'Sec. 5', 'Article 5', 'Clause 5' patterns.
        """
        patterns = [
            r'\bSection\s+([0-9A-Za-z\-\.\(\/\)]+)', 
            r'\bSec\.?\s*([0-9A-Za-z\-\.\(\/\)]+)',
            r'\bs\.\s*([0-9A-Za-z\-\.\(\/\)]+)',
            r'\bArticle\s+([0-9A-Za-z\-\.\(\/\)]+)',
            r'\bClause\s+([0-9A-Za-z\-\.\(\/\)]+)'
        ]
        for p in patterns:
            m = re.search(p, text, flags=re.IGNORECASE)
            if m:
                return m.group(0).strip()
        return "Unknown"

    def extract_chapter(self, text: str) -> str:
        m = re.search(r'\bChapter\s+([IVXLCDM0-9A-Za-z\-]+)', text, flags=re.IGNORECASE)
        return m.group(0) if m else "Unknown"

    def detect_source_page(self, text: str) -> int:
        """
        Heuristic to detect page marker 'Page 12', 'Pg. 12', 'p.12'. Return first number found near 'page' tokens.
        """
        m = re.search(r'\bPage[:\s]*([0-9]{1,4})\b', text, flags=re.IGNORECASE)
        if m:
            return int(m.group(1))
        m = re.search(r'\bPg\.?\s*([0-9]{1,4})\b', text, flags=re.IGNORECASE)
        if m:
            return int(m.group(1))
        m = re.search(r'\bp\.\s*([0-9]{1,4})\b', text, flags=re.IGNORECASE)
        if m:
            return int(m.group(1))
        return "Unknown"

    def detect_property_relevance(self, text: str) -> bool:
        """
        Simple keyword-based classifier: returns True if chunk contains strong property/legal keywords.
        """
        keywords_signal = [
            "land", "property", "ownership", "possession", "mutation", "registry", "allotment",
            "lease", "rent", "evict", "eviction", "encroachment", "acquisition", "compensation",
            "title", "transfer", "mutation", "khasra", "deed", "registration", "conveyance", "possession",
            "tenant", "landlord", "plot", "allottee"
        ]
        text_low = text.lower()
        score = sum(1 for k in keywords_signal if k in text_low)
        return score >= 1  # 1 or more signals -> property-related (adjust threshold if needed)

    def extract_keywords(self, text: str, top_k: int = 8) -> List[str]:
        """Lightweight keyword extraction: top frequent non-stopwords of length>=4"""
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        filtered = [w for w in words if w not in self.STOPWORDS]
        counter = Counter(filtered)
        most = [w for w, _ in counter.most_common(top_k)]
        return most
    
    def estimate_token_count(self, text: str) -> int:
        """Approximate token count (simple heuristic: 1 token ~ 0.75 words)."""
        words = len(text.split())
        return math.ceil(words / 0.75) if words > 0 else 0

    def initialize_gemini(self):
        """Initialize Gemini client"""
        try:
            import google.generativeai as genai

            if not self.google_api_key:
                raise ValueError("No Google API key provided")
            genai.configure(api_key=self.google_api_key)
            self.client = genai
            print("Gemini client initialized successfully!")

        except ImportError:
            raise ImportError("google.generativeai not installed. Run: pip install google-generativeai")
        except Exception as e:
            raise Exception(f"Error initializing Gemini: {e}")

    def initialize_pinecone(self, dimension):
        """Initialize Pinecone connection for Gemini embeddings"""
        try:
            from pinecone import Pinecone, ServerlessSpec
            
            if not self.pinecone_api_key:
                raise ValueError("No Pinecone API key provided")
            
            # Initialize Pinecone client
            self.pc = Pinecone(api_key=self.pinecone_api_key)
            
            # Check if index exists, create if not
            existing_indexes = self.pc.list_indexes()
            index_names = [idx.name for idx in existing_indexes.indexes] if hasattr(existing_indexes, 'indexes') else []
            
            if self.index_name not in index_names:
                self.pc.create_index(
                    name=self.index_name,
                    dimension=dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                print("Index created successfully!")
                time.sleep(30)
            else:
                print(f"Index '{self.index_name}' already exists")
            
            # Connect to index
            self.index = self.pc.Index(self.index_name)
            print("Pinecone initialized successfully!")
            
        except ImportError:
            raise ImportError("pinecone not installed. Run: pip install pinecone")
        except Exception as e:
            raise Exception(f"Error initializing Pinecone: {e}")
        
    def get_file_chunks(self, filename_year_map: Dict[str,int]=None) -> Tuple[List[Chunk], List[Dict]]:
        """
        Fetch raw DB chunk objects and build processed chunk dicts.
        Returns: (all_chunks (DB objects), processed_chunks (list of dict))
        """
        all_chunks = []
        skip = 0
        limit = 100

        while True:
            chunk_page = get_all_chunks(skip=skip, limit=limit)
            if not chunk_page:
                break
            
            all_chunks.extend(chunk_page)
            skip += limit

        print(f"Total chunks fetched: {len(all_chunks)}")
        processed = []
        for chunk in all_chunks:
        
            # compute file_name: remove chunk_ prefix and trailing _NNN.txt
            raw_name = chunk.chunk_name
            file_name_guess = re.sub(r'^chunk[_\-]*', '', raw_name, flags=re.IGNORECASE)
            file_name_guess = re.sub(r'_\d{1,4}\.txt$', '', file_name_guess)
            file_name_guess = file_name_guess.strip()
            law_name, law_year, law_type = self.parse_filename_for_law(file_name_guess, filename_year_map)

            text = chunk.chunk_data or ""
            token_count = self.estimate_token_count(text)
            is_property = self.detect_property_relevance(text)
            keywords = self.extract_keywords(text, top_k=8)
            section = self.extract_section_number(text)
            chapter = self.extract_chapter(text)
            source_page = self.detect_source_page(text)        
            
            processed_chunk = {
                "chunk_id": chunk.id,
                "chunk_name": chunk.chunk_name,
                "file_name": law_name,
                "law_name": law_name,
                "law_year": law_year,
                "law_type": law_type,
                "section_number": section,
                "chapter": chapter,
                "text": text,
                "word_count": len(text.split()),
                "char_count": len(text),
                "token_count": token_count,
                "is_property_related": is_property,
                "keywords": keywords,
                "source_page": source_page,
                # embedding placeholders
                "embedding": None,
                "embedding_dim": None,
                "embedding_model": None,
                "processed_at": None
            }
            processed.append(processed_chunk)

        return all_chunks, processed

    def test_connections(self):
        try:
            import google.generativeai as genai
            print("Testing Gemini connection...")

            genai.configure(api_key=self.google_api_key)

            # Test with a small request
            response = genai.embed_content(
                model=self.embedding_model,
                content="Test connection"
            )
            print(f"Gemini connection successful!")
            try:
                from pinecone import Pinecone
                print("Testing Pinecone connection for Gemini...")
                
                pc = Pinecone(api_key=self.pinecone_api_key)
                indexes = pc.list_indexes()
                index_names = [idx.name for idx in indexes.indexes] if hasattr(indexes, 'indexes') else []
                print(f"Pinecone connection successful!")
                return True
                
            except Exception as e:
                print(f"Pinecone connection failed: {e}")
                return False
        except Exception as e:
            print(f"Gemini connection failed: {e}")
            return False

    def process_chunks(self, all_chunks: List[Chunk], chunks: List[Dict], embed_batch_size: int = 30, pinecone_batch_size: int = 50):
        """
        Full pipeline: process in batches:
        - generate_embeddings (embed_batch_size per batch for Gemini)
        - store_in_pinecone (pinecone_batch_size per upsert)
        - mark DB rows as embedded via self.update_chunk_as_embedded()
        """

        total = len(chunks)
        print(f"Starting processing of {total} chunks (embed_batch_size={embed_batch_size})")
        all_processed = []
        for start in range(0, total, embed_batch_size):
            end = min(start + embed_batch_size, total)
            batch = chunks[start:end]
            batch_all_chunks = all_chunks[start:end]

            print(f"\n=== Processing batch {start // embed_batch_size + 1}: chunks {start}..{end-1} ===")
            try:
                # Generate embeddings
                chunks_with_embeddings = self.generate_embeddings(batch)
                if not chunks_with_embeddings:
                    print("No embeddings returned for batch — skipping.")
                    continue

                # Update embedding-related metadata in each chunk_with_embeddings
                ts_iso = datetime.now().isoformat()
                for c in chunks_with_embeddings:
                    c.setdefault('embedding_model', getattr(self, 'embedding_model', 'gemini-embedding-001'))
                    c.setdefault('processed_at', ts_iso)
                    c.setdefault('embedding_dim', len(c.get('embedding', [])))
                
                # 2) Store in Pinecone (use global offset to create stable IDs)
                global_offset = start  # used to generate stable vector ids
                stored = self.store_in_pinecone(chunks_with_embeddings, batch_size=pinecone_batch_size, global_offset=global_offset)
                if stored:
                    # mark those DB chunks as embedded (use your provided function)
                    try:
                        self.update_chunk_as_embedded(batch_all_chunks)
                    except Exception as e:
                        print(f"Warning: update_chunk_as_embedded failed: {e}")

                all_processed.extend(chunks_with_embeddings)
            except Exception as e:
                print(f"Error in processing batch {start//embed_batch_size + 1}: {e}")
                continue
        
        print(f"\nProcessing complete. Total processed/embedded: {len(all_processed)}")
        return all_processed
    
    def generate_embeddings(self, chunks: List[Dict]) -> List[Dict]:
        """
        Generate embeddings using self.client (Gemini). Returns list of chunk dicts with 'embedding' field.
        Respects a small cooldown and handles 429s.
        """
        if not hasattr(self, 'client') or not self.client:
            self.initialize_gemini()
        
        chunks_with_embeddings = []
        cooldown_after = 20       # number of chunks before cooldown
        cooldown_seconds = 30
        per_request_delay = 0.5
        
        for i, chunk in enumerate(chunks):
            # local progress print
            print(f"Embedding chunk {i+1}/{len(chunks)} (chunk_id={chunk.get('chunk_id')})")

            if i > 0 and i % cooldown_after == 0:
                print(f"\n=== Cooling down for {cooldown_seconds} inside batch ===")
                time.sleep(cooldown_seconds)
            try:
                # Call Gemini API
                response = self.client.embed_content(
                    model=self.embedding_model,
                    content=chunk['text']
                )
                
                # response may vary shape; try common keys
                embedding = None
                if isinstance(response, dict):
                    embedding = response.get('embedding') or response.get('data', [{}])[0].get('embedding')
                else:
                    # if response object from SDK has .embedding
                    embedding = getattr(response, 'embedding', None)

                if not embedding:
                    raise Exception("No embedding returned by Gemini client")
                
                # Add embedding to chunk
                chunk_with_embedding = dict(chunk)  # shallow copy
                chunk_with_embedding['embedding'] = embedding
                chunk_with_embedding['embedding_dim'] = len(embedding)
                chunk_with_embedding['embedding_model'] = self.embedding_model
                chunk_with_embedding['processed_at'] = datetime.now().isoformat()
                
                chunks_with_embeddings.append(chunk_with_embedding)
                
                # Rate limiting - be nice to the API
                time.sleep(per_request_delay)
                
            except Exception as e:
                err_str = str(e)
                print(f"Error processing chunk {i+1}: {err_str}")
                if "429" in err_str or "quota" in err_str.lower():
                    print("Rate limit hit — backing off 60 seconds...")
                    time.sleep(60)
                # skip this chunk but continue with others
                continue

        print(f"Gemini embeddings generated successfully! ({len(chunks_with_embeddings)}/{len(chunks)} chunks)")
        return chunks_with_embeddings
    
    def store_in_pinecone(self, chunks_with_embeddings: List[Dict], batch_size: int =50, global_offset: int = 300) -> bool:
        """
        Store embeddings in Pinecone with the full Sindh legal metadata.
        - global_offset: used to generate stable unique vector ids across batches
        """
        if not chunks_with_embeddings:
            return False
    
        # determine embedding dimension
        first = chunks_with_embeddings[0]
        actual_dimension = first.get('embedding_dim') or len(first.get('embedding', []))
        if not actual_dimension:
            actual_dimension = getattr(self, 'embedding_dimension', None) or len(first.get('embedding', []))
        
        # ensure Pinecone index initialized (call only once ideally)
        try:
            self.initialize_pinecone(dimension=actual_dimension)
        except Exception:
            # assume already initialized or self.initialize_pinecone handles re-init safely
            pass

        print(f"Storing {len(chunks_with_embeddings)} Gemini vectors in Pinecone...")

        vectors = []
        now_ts = time.time()
        for i, chunk in enumerate(chunks_with_embeddings):
            # stable unique id: sanitized file name + DB chunk id (or global offset + i)
            chunk_id = str(chunk.get('chunk_id') or (global_offset + i) or uuid.uuid4().hex)
            file_name = chunk.get('file_name', 'unknown')
            safe_name = self.sanitize_filename(file_name)
            vector_id = f"{safe_name}_chunk_{chunk_id}"

            metadata = {
                "chunk_id": chunk_id,
                "chunk_name": chunk.get('chunk_name'),
                "file_name": chunk.get('file_name', file_name),
                "law_name": chunk.get('law_name'),
                "law_year": chunk.get('law_year'),
                "law_type": chunk.get('law_type'),
                "section_number": chunk.get('section_number'),
                "chapter": chunk.get('chapter'),
                "jurisdiction": "Sindh",
                "domain": "Real Estate",
                "is_property_related": bool(chunk.get('is_property_related', False)),
                "keywords": chunk.get('keywords', []),
                "text_preview": (chunk.get('text') or "")[:300],
                "full_text": chunk.get('text', ""),
                "token_count": chunk.get('token_count', self.estimate_token_count(chunk.get('text',''))),
                "source_page": chunk.get('source_page'),
                "embedding_model": chunk.get('embedding_model', getattr(self, 'embedding_model', 'gemini')),
                "processed_at": chunk.get('processed_at', datetime.now().isoformat()),
                "timestamp": now_ts,
                "chunk_index": global_offset + i
            }
            
            vectors.append({
                "id": vector_id,
                "values": chunk.get('embedding'),
                "metadata": metadata
            })
        
        total_vectors = len(vectors)
        successful_vectors = 0

        safe_batch = max(1, min(batch_size, 100))  # clamp to 1..100

        # Smaller batch size for API stability
        for start in range(0, total_vectors, safe_batch):
            batch = vectors[start:start + safe_batch]
            try:
                self.index.upsert(vectors=batch)
                successful_vectors += len(batch)
                print(f"Upserted batch {start//safe_batch + 1}/{(total_vectors-1)//safe_batch + 1} - {len(batch)} vectors")
            except Exception as e:
                print(f"Failed to upsert batch {start//safe_batch + 1}: {e}")
                # on failure, attempt single-upsert fallback to identify bad vector
                for vec in batch:
                    try:
                        self.index.upsert(vectors=[vec])
                        successful += 1
                    except Exception as ee:
                        print(f"Failed single upsert for id {vec['id']}: {ee}")
                continue
        
        print(f"Successfully stored {successful_vectors}/{total_vectors} vectors in Pinecone!")
        return True

    def get_embedding_dimension(self):
        """Get the actual embedding dimension by testing with a small query"""
        if not self.client:
            self.initialize_gemini()
        try:
            # Test with a small query to get actual dimension
            import google.generativeai as genai
            test_response = genai.embed_content(
                model=self.embedding_model,
                content="test"
            )
            actual_dimension = len(test_response['embedding'])
            return actual_dimension
        except Exception as e:
            print(f"Error detecting dimension, using default: {e}")
            return 3072  # Fallback dimension
    
    def update_chunk_as_embedded(self, chunks: List[Chunk]):
        """Mark chunk as embedded in the database"""
        try:
            for chunk in chunks:
                mark_chunk_embedded(chunk.id)
            print(f"Marked {len(chunks)} chunks as embedded in the database.")
        except Exception as e:
            print(f"Error marking chunks as embedded: {e}")