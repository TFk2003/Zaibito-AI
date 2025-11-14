from code.logic.chunk_creator import ChunkCreator
import os

def working():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    directory_path = os.path.join(script_dir, "data")
    directory_path = os.path.normpath(directory_path)
    chunk_creator = ChunkCreator(chunk_size=400, chunk_overlap=50)
    directory_files = chunk_creator.get_directory_files(directory_path)
    stored_files = chunk_creator.stored_files()

    stored_filenames = [file.file_name for file in stored_files]

    for pdf_file in directory_files:
        filename = os.path.basename(pdf_file)
        file_id = chunk_creator.get_file_id_by_name(file_name=filename)

        if filename in stored_filenames:
            file = chunk_creator.get_file_by_id(file_id=file_id)
            if file.chunked:
                continue
        
        print(f"Processing file: {pdf_file}")
        if filename not in stored_filenames:
            chunk_creator.add_law_file(pdf_path=pdf_file)
        chunks = chunk_creator.process_pdf_to_chunks(
            pdf_file,
            chunk_size=chunk_creator.chunk_size,
            chunk_overlap=chunk_creator.chunk_overlap,
            use_nltk=True
        )
        if chunks:
            base_filename = "chunk_" + filename.removesuffix(".pdf")
            file_id = chunk_creator.get_file_id_by_name(file_name=filename)
            chunk_creator.save_chunks(chunks, file_id=file_id, base_filename=base_filename)
            chunk_creator.mark_file_as_chunked(file_id=file_id)
        else:
            print("No chunks created.")

if __name__ == "__main__":
    working()