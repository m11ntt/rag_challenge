import os
import psycopg2
import torch
from openai import OpenAI
import numpy as np
from dotenv import load_dotenv
from FlagEmbedding import BGEM3FlagModel

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DB_PARAMS = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": int(os.getenv("DB_PORT", 5432))
}

device = "cuda" if torch.cuda.is_available() else "cpu"
embed_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, device=device)

client = OpenAI(api_key=OPENAI_API_KEY)

def get_embedding(query: str):
    with torch.no_grad():
        output = embed_model.encode([query], max_length=8192)
        return output["dense_vecs"][0]

def vector_search(conn, question: str, k: int = 5):
    embedding = get_embedding(question)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, filename, page_number, content, 1 - (embedding <=> %s::vector) AS score
            FROM chunks_table
            ORDER BY score DESC
            LIMIT %s;
            """,
            (embedding.tolist(), k),
        )
        results = cur.fetchall()
    return results

def keyword_search(conn, question: str, k: int = 10):
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, filename, page_number, content,
                   ts_rank_cd(tsv, plainto_tsquery('russian', %s)) AS score
            FROM chunks_table
            WHERE tsv @@ plainto_tsquery('russian', %s)
            ORDER BY score DESC
            LIMIT %s;
            """, (question, question, k)
        )
        results = cur.fetchall()
    return results

def hybrid_retriever_rrf(conn, question: str, top_k: int = 5, k: int = 60):
    keyword_results = keyword_search(conn, question, k=20)
    vector_results = vector_search(conn, question, k=20)

    kw_ranks = {doc[0]: rank for rank, doc in enumerate(keyword_results, start=1)}
    vec_ranks = {doc[0]: rank for rank, doc in enumerate(vector_results, start=1)}

    combined = {}

    for doc_id, rank in kw_ranks.items():
        combined.setdefault(doc_id, {"data": None, "score": 0})
        combined[doc_id]["data"] = keyword_results[rank - 1]
        combined[doc_id]["score"] += 1 / (k + rank)

    for doc_id, rank in vec_ranks.items():
        combined.setdefault(doc_id, {"data": None, "score": 0})
        combined[doc_id]["data"] = vector_results[rank - 1]
        combined[doc_id]["score"] += 1 / (k + rank)

    results = sorted(
        [(doc["data"][0], doc["data"][1], doc["data"][2], doc["data"][3], doc["score"]) for doc in combined.values()],
        key=lambda x: x[4],
        reverse=True,
    )

    return results[:top_k]

def get_page_text(conn, filename, page_number):
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT content
            FROM chunks_table
            WHERE filename = %s AND page_number = %s
            ORDER BY id
            """,
            (filename, page_number),
        )
        rows = cur.fetchall()
    return "\n".join([r[0] for r in rows])

def answer_question(question: str, answer_type: str):
    with psycopg2.connect(**DB_PARAMS) as conn:
        retrieved_chunks = hybrid_retriever_rrf(conn, question, top_k=10)

        seen_pages = set()
        page_contexts = []
        for _, fname, pg, _, _ in retrieved_chunks:
            page_key = (fname, pg)
            if page_key not in seen_pages:
                seen_pages.add(page_key)
                full_page_text = get_page_text(conn, fname, pg)
                page_contexts.append(f"[{fname} p.{pg}] {full_page_text}")

    context_text = "\n\n".join(page_contexts)

    prompt = f"""
You are a helpful assistant. Use only the provided context to answer the question. Do not use any outside knowledge or assumptions.
Requirements for the answer:
- Answer only in Russian
- Return only the exact answer, based on the stated data type (int/float/str)
- Provide a direct and accurate answer strictly based on the context
- Include the documents name and the pages number where the information was found
- If the answer cannot be found in the context, clearly state: “Информация не предоставлена в контексте”
- Keep the answer concise, factual, and faithful to the source

Output Format:
<your answer> | <document1 name>, <page1 number>; <document2 name>, <page2 number>, ...

Context:
{context_text}

Answer type:
{answer_type}

Question: {question}
Answer:
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a RAG assistant."},
            {"role": "user", "content": prompt}
        ],
    )

    return response.choices[0].message.content


question = '''
        Какой организацией представлен АО «НК Қазақстан Темір Жолы» в Венгрии?
        '''
answer_type = "str"
print(f'\n\n{answer_question(question, answer_type)}')