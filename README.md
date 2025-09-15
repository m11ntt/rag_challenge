# RAG project

## Overview
This project implements a Retrieval-Augmented Generation (RAG) pipeline for document-based question answering.
It combines vector search (semantic embeddings) and keyword search (full-text) to deliver context-rich responses from structured and unstructured documents (scanned PDFs, Excel sheets, company reports, etc.).

---

## Tools
- Python 3.x
- PostgreSQL + pgvector

---

## Workflow
1. **Data Preprocessing:** 
    - Convert Excel files into PDF for simpler processing
    - Extract text from scanned PDFs and Excel tables using [Marker OCR](https://github.com/datalab-to/marker)
    - Extract text from other PDFs (e.g., company's reports) using [LlamaParse](https://www.llamaindex.ai/llamaparse) or [PyMuPDF4LLM](https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/)
    - Converted rows in well-structured tables clearer (LLM-friendly) chunks: 
    ```yaml
    id: ... 
    Name: ...
    Date of Birth: ...
    ...
    ```
2. **Database setup:** 
    - Split document pages into chunks
    - Retrieve embeddings using [BAAI/bge-m3 model](https://huggingface.co/BAAI/bge-m3)
    - Store data in a PostgreSQL database (using both vector and tsvector columns)
3. **RAG Pipeline:**
    - Implement hybrid retrieval with vector + keyword search and RRF (Reciprocal Rank Fusion)
    - Extend context by including full page text for each relevant chunk
4. **Asynchronous Pipeline:** 
    - Update the pipeline for processing larger Q&A datasets

---

## Installation 
```bash
git clone https://github.com/m11ntt/rag_challenge.git
cd rag_challenge
pip install -r requirements.txt
```
pgvector extension:
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```
tsvector column
```sql
ALTER TABLE chunks_table ADD COLUMN tsv tsvector;

UPDATE chunks_table
SET tsv = to_tsvector('russian', content);
```