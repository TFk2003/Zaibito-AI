from model_code.models.law_file import LawFile
from sqlalchemy.orm import Session
from typing import List

def add_file(db: Session, file_name: str, file_data: bytes, chunked: bool = False) -> LawFile:

    new_file = LawFile(
        file_name=file_name,
        file_data=file_data,
        chunked=chunked
    )
    db.add(new_file)
    db.commit()
    db.refresh(new_file)
    return new_file

def update_file_chunked_status(db: Session, file_id: int, chunked: bool) -> LawFile:
    file_record = db.query(LawFile).filter(LawFile.id == file_id).first()
    if file_record:
        file_record.chunked = chunked
        db.commit()
        db.refresh(file_record)
    return file_record

def get_files(db: Session, skip: int = 0, limit: int = 100) -> List[LawFile]:
    return db.query(LawFile).offset(skip).limit(limit).all()

def get_file_id_by_name(db: Session, file_name: str) -> int:
    file_record = db.query(LawFile).filter(LawFile.file_name == file_name).first()
    return file_record.id if file_record else None

def get_file_by_id(db: Session, file_id: int) -> LawFile:
    return db.query(LawFile).filter(LawFile.id == file_id).first()