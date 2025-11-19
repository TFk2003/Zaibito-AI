from model_code.models.chunk import Chunk
from sqlalchemy.orm import Session
from typing import List

def create_chunk(db: Session, chunk_name, chunk_data, file_id) -> Chunk:
    chunk_record = Chunk(chunk_name=chunk_name, chunk_data=chunk_data, file_id=file_id, embedded=False)
    db.add(chunk_record)
    db.commit()
    db.refresh(chunk_record)
    return chunk_record

def get_unembedded_chunks(db: Session, skip: int = 0, limit: int = 100) -> List[Chunk]:
    return db.query(Chunk).filter(Chunk.embedded == False).offset(skip).limit(limit).all()

def get_embedded_chunks(db: Session, skip: int = 0, limit: int = 100) -> List[Chunk]:
    return db.query(Chunk).filter(Chunk.embedded == True).offset(skip).limit(limit).all()

def mark_chunk_as_embedded(db: Session, chunk_id: int) -> Chunk:
    chunk_record = db.query(Chunk).filter(Chunk.id == chunk_id).first()
    if chunk_record:
        chunk_record.embedded = True
        db.commit()
        db.refresh(chunk_record)
    return chunk_record

def get_chunk_by_id(db: Session, chunk_id: int) -> Chunk:
    return db.query(Chunk).filter(Chunk.id == chunk_id).first()