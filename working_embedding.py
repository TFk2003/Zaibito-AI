from model_code.logic.embedding_creator import EmbeddingCreator

def WorkingEmbedding():
    try:
        print("=== GEMINI EMBEDDING PIPELINE ===")
        embedding_creator = EmbeddingCreator()
        embedding_creator.test_connections()
        chunks = embedding_creator.get_file_chunks()
        processed_chunks = embedding_creator.process_chunks(chunks)
        if processed_chunks:
            print(f"Processed {len(processed_chunks)} chunks with embeddings.")
    except Exception as e:
        print(f"Error in embedding pipeline: {e}")

if __name__ == "__main__":
    WorkingEmbedding()
