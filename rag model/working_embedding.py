from model_code.logic.embedding_creator import EmbeddingCreator
import json
def WorkingEmbedding():
    try:
        print("=== GEMINI EMBEDDING PIPELINE ===")
        embedding_creator = EmbeddingCreator()
        embedding_creator.test_connections()

        # Load your document_year_mapping.json
        with open("document_year_mapping.json", "r", encoding="utf-8") as f:
            filename_year_map = json.load(f)

        all_chunks, chunks = embedding_creator.get_file_chunks(filename_year_map=filename_year_map)
        if not chunks:
            print("No unembedded chunks found.")
            return
        processed_chunks = embedding_creator.process_chunks(all_chunks, chunks)
        if processed_chunks:
            print(f"Processed {len(processed_chunks)} chunks with embeddings.")
    except Exception as e:
        print(f"Error in embedding pipeline: {e}")

if __name__ == "__main__":
    WorkingEmbedding()