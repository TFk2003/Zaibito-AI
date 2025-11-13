from code.service.chunk import create_chunk
from code.core.database import SessionLocal

def create_new_chunk(chunk_name, chunk_data, file_id):
    chunk = create_chunk(db=SessionLocal(), chunk_name=chunk_name, chunk_data=chunk_data, file_id=file_id)
    print(f"Created chunk: {chunk_name} for file ID: {file_id}")
    return chunk