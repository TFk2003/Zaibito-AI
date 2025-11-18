from model_code.service.chunk import create_chunk, get_embedded_chunks, get_unembedded_chunks, mark_chunk_as_embedded
from model_code.core.database import SessionLocal

def create_new_chunk(chunk_name, chunk_data, file_id):
    with SessionLocal() as db:
        chunk = create_chunk(db=db, chunk_name=chunk_name, chunk_data=chunk_data, file_id=file_id)
        return chunk
    
def get_all_chunks(skip: int = 0, limit: int = 100):
    with SessionLocal() as db:
        chunks = get_unembedded_chunks(db=db, skip=skip, limit=limit)
        return chunks
    
def mark_chunk_embedded(chunk_id: int):
    with SessionLocal() as db:
        chunk = mark_chunk_as_embedded(db=db, chunk_id=chunk_id)
        return chunk
    
def get_all_embedded_chunks(skip: int = 0, limit: int = 100):
    with SessionLocal() as db:
        chunks = get_embedded_chunks(db=db, skip=skip, limit=limit)
        return chunks