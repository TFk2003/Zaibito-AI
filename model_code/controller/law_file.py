from model_code.service.law_file import add_file, get_file_by_id, get_files, update_file_chunked_status, get_file_id_by_name
from model_code.core.database import SessionLocal
import os
class LawFileController:
    
    def add_law_file(self, pdf_path: str):
        with SessionLocal() as db:
            with open(pdf_path, "rb") as file:
                file_data = file.read()
            file_name = os.path.basename(pdf_path)
            add_file(db=db, file_name=file_name, file_data=file_data)
    
    def list_law_files(self):
        with SessionLocal() as db:
            files = get_files(db=db)
        return files

    def update_law_file_chunked_status(self, file_id: int, chunked: bool):
        with SessionLocal() as db:
            updated_file = update_file_chunked_status(db=db, file_id=file_id, chunked=chunked)
            return updated_file
    
    def get_file_id_by_name(self, file_name: str) -> int:
        with SessionLocal() as db:
            file_id = get_file_id_by_name(db=db, file_name=file_name)
            return file_id
    
    def get_file_by_id(self, file_id: int):
        with SessionLocal() as db:
            file = get_file_by_id(db=db, file_id=file_id)
            return file