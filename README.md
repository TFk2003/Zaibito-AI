# ZABITO AI – Real Estate Legal Assistant (Sindh)

A Retrieval-Augmented Generation (RAG)–based legal assistant specialized in real estate laws of Sindh, Pakistan.

This project extracts text from law PDFs, chunks them, generates embeddings using Gemini, stores vectors in Pinecone, and provides an intelligent legal assistant interface.

---

## 🚀 Setup Guide

Follow the steps below to fully set up and run the system.

---

## 1. Install Tesseract & Poppler

Download **Tesseract OCR** and **Poppler** from the links provided in `requirements.txt`.

After installation:

- Add the Tesseract installation folder to your **PATH** environment variable.
- Add the Poppler `bin` directory to your **PATH** environment variable.

---

## 2. Install Dependencies

```bash
cd rag_model
pip install -r requirements.txt
```

---

## 3. Create .env File

Create the file:

```bash
project/rag_model/model_code/.env
```

Add the following environment variables:

```bash
DATABASE_URL=your_database_url
GOOGLE_API_KEY=your_google_api_key
PINECONE_API_KEY=your_pinecone_api_key
EMBEDDING_MODEL=your_gemini_embedding_model
INDEX_NAME=your_pinecone_index_name
```

---

## 4. Add Your Law Files

Place all your PDF law documents inside:

```bash
project/rag_model/data/
```

Only .pdf files are supported.

---

## 5. Run Chunking Script

This extracts text, cleans it, and generates chunks:

```bash
project/rag_model/model_code/working.py
```

This will populate the database with file chunks.

---

## 6. Add Document-Year Mapping (Temporary)

Open:

```bash
project/rag_model/responses/document_year_mapping.json
```

Append mappings like:

{
  "Building Bye-Laws 2007": 2007,
  "Karachi Building & Town Planning Regulations 2002": 2002
}
(Automatic extraction coming soon.)

---

## 7. Run Embedding Script
