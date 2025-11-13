from code.models.chunk import Chunk
from sqlalchemy.orm import Session
from typing import List

def create_chunk(db: Session, chunk_name, chunk_data, file_id) -> Chunk:
    chunk_record = Chunk(chunk_name=chunk_name, chunk_data=chunk_data, file_id=file_id)
    db.add(chunk_record)
    db.commit()
    db.refresh(chunk_record)
    return chunk_record

def get_chunks(db: Session) -> List[Chunk]:
    return db.query(Chunk).all()