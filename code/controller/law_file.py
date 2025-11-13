from code.service.law_file import add_file, get_files, update_file_chunked_status, get_file_id_by_name
from code.core.database import SessionLocal
class LawFileController:
    def __init__(self):
        self.db = SessionLocal()
    def add_law_file(self, pdf_path: str):
        with open(pdf_path, "rb") as file:
            file_data = file.read()
        file_name = pdf_path.split("/")[-1]
        add_file(db=self.db, file_name=file_name, file_data=file_data)
        print(f"Added law file: {file_name}")

    def list_law_files(self):
        files = get_files(db=self.db)
        return files

    def update_law_file_chunked_status(self, file_id: int, chunked: bool):
        updated_file = update_file_chunked_status(db=self.db, file_id=file_id, chunked=chunked)
        return updated_file
    def get_file_id_by_name(self, file_name: str) -> int:
        return get_file_id_by_name(db=self.db, file_name=file_name)