from code.service.chunk import create_chunk
from code.core.database import SessionLocal

def create_new_chunk(chunk_name, chunk_data, file_id):
    with SessionLocal() as db:
        chunk = create_chunk(db=db, chunk_name=chunk_name, chunk_data=chunk_data, file_id=file_id)
        return chunk