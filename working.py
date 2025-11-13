from code.logic.chunk_creator import ChunkCreator
import os

def working():
    directory_path = "F:\\E\\classroom\\SEMESTER 7\\FYP\\Project\\data"
    chunk_creator = ChunkCreator(chunk_size=300, chunk_overlap=30)
    directory_files = chunk_creator.get_directory_files(directory_path)
    stored_files = chunk_creator.stored_files()

    stored_filenames = [file.file_name for file in stored_files]

    for pdf_file in directory_files:
        filename_only = os.path.basename(pdf_file)
        if(filename_only not in stored_filenames):
            print(f"Processing file: {pdf_file}")
            chunk_creator.add_law_file(pdf_path=pdf_file)
            chunks = chunk_creator.process_pdf_to_chunks(
                pdf_file,
                chunk_size=chunk_creator.chunk_size,
                chunk_overlap=chunk_creator.chunk_overlap,
                use_nltk=True
            )
        
        if chunks:
            base_filename = "chunk_" + os.path.splitext(filename_only)[0]
            file_id = chunk_creator.get_file_id_by_name(file_name=filename_only)
            chunk_creator.save_chunks(chunks, file_id=file_id, base_filename=base_filename)
            chunk_creator.mark_file_as_chunked(file_id=file_id)
        else:
            print("No chunks created.")

if __name__ == "__main__":
    working()