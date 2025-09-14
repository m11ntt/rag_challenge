import os
import re
import json
import torch
import asyncio
import asyncpg
import logging
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from openai import AsyncOpenAI
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from FlagEmbedding import BGEM3FlagModel


# CONFIG
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DB_PARAMS = {
    "host": os.getenv("DB_HOST"),
    "port": int(os.getenv("DB_PORT", 5432)),
    "database": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
}

client = AsyncOpenAI(api_key=OPENAI_API_KEY)
OPENAI_CONCURRENCY = 2

SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent
INPUT_PATH = PARENT_DIR / "input" / "questions_test.xlsx"

OUTPUT_DIR = PARENT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
AUTOSAVE_PATH = OUTPUT_DIR / "answers_autosave.json"
OUTPUT_PATH = OUTPUT_DIR / "answers.json"


# LOGGING
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


# EMBEDDING
device = "cuda" if torch.cuda.is_available() else "cpu"
model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, device=device)

def get_embedding(query: str):
    with torch.no_grad():
        output = model.encode([query], max_length=8192)
        return output["dense_vecs"][0]


# DB SEARCH
async def keyword_search(conn, question: str, k: int = 10):
    rows = await conn.fetch(
        """
        SELECT id, filename, page_number, content,
            ts_rank_cd(tsv, plainto_tsquery('russian', $1)) AS score
        FROM chunks_table
        WHERE tsv @@ plainto_tsquery('russian', $1)
        ORDER BY score DESC
        LIMIT $2;
        """,
        question,
        k,
    )
    return rows

async def vector_search(conn, question: str, k: int = 10):
    embedding = get_embedding(question)
    embedding = embedding.tolist()
    rows = await conn.fetch(
        """
        SELECT id, filename, page_number, content,
            1 - (embedding <=> $1::vector) AS score
        FROM chunks_table
        ORDER BY score DESC
        LIMIT $2;
        """,
        str(embedding),
        k,
    )
    return rows


# HYBRID RETRIEVER + RRF
async def hybrid_retriever_rrf(conn, question: str, top_k: int = 5, k: int = 60):
    keyword_results = await keyword_search(conn, question, k=15)
    vector_results = await vector_search(conn, question, k=15)

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
        [
            {
                "id": doc["data"][0],
                "filename": doc["data"][1],
                "page_number": doc["data"][2],
                "content": doc["data"][3],
                "score": doc["score"]
            }
            for doc in combined.values()
        ],
        key=lambda x: x["score"],
        reverse=True,
    )

    return results[:top_k]


# GET CONTEXT
async def get_page_text(conn, filename, page_number):
    rows = await conn.fetch(
        """
        SELECT content
        FROM chunks_table
        WHERE filename = $1 AND page_number = $2
        ORDER BY id
        """,
        filename,
        page_number,
    )
    return "\n".join([r["content"] for r in rows])


# ANSWER GENERATION
async def generate_answer(question: str, context_text: str, answer_type: str):
    prompt = f"""
Ты полезный ассистент. Используй только предоставленный контекст для ответа.
Никаких внешних знаний и догадок.

Требования:
- Отвечай только на русском языке.
- Дай только точный ответ указанного типа (int/float/str).
- Ответ должен включать названия документов и номера страниц, которые ты использовал для ответа.
- Если информации нет в контексте, ответь: "-" либо null и не добавляй названия документов и номера страниц в ответ.
- Будь кратким, фактическим и основанным исключительно на контексте.
- Отвечай строго в указанном формате.

Формат ответа:
<твой ответ>| <название документа1>, <номер страницы1>; <название документа2>, <номер страницы2>; ...

Контекст:
{context_text}

Тип ответа: {answer_type}

Вопрос: {question}
Ответ:
    """

    for attempt in range(5):
        try:
            response = await client.chat.completions.create(
                model="gpt-5-mini",
                messages=[
                    {"role": "system", "content": "Ты - полезный помощник в ответах на вопросы."},
                    {"role": "user", "content": prompt}
                ],
            )
            return response.choices[0].message.content
        except Exception as e:
            wait = 2 ** attempt
            logging.warning(f"Error in LLM call: {e}. Retrying in {wait}s...")
            await asyncio.sleep(wait)


# PIPELINE
async def answer_question(pool, qid, question, answer_type, semaphore):
    async with pool.acquire() as conn:
        async with semaphore:
            logging.info(f"[Q{qid}] Retrieving context...")
            retrieved_chunks = await hybrid_retriever_rrf(conn, question, top_k=10)

            seen_pages = set()
            page_contexts = []
            for row in retrieved_chunks:
                page_key = (row["filename"], row["page_number"])
                if page_key not in seen_pages:
                    seen_pages.add(page_key)
                    page_text = await get_page_text(conn, row["filename"], row["page_number"])
                    page_contexts.append(f"[{row['filename']} p.{row['page_number']}] {page_text}")

            context_text = "\n\n".join(page_contexts)
            logging.info(f"[Q{qid}] Context: {', '.join([f'{fname} p.{pg}' for fname, pg in seen_pages])}")
            logging.info(f"[Q{qid}] Sending to LLM...")

            answer_text = await generate_answer(question, context_text, answer_type)

            raw_answer = None
            relevant_chunks = []

            if answer_text:
                parts = answer_text.split("|", maxsplit=1)
                raw_answer = parts[0].strip() if parts[0].strip() not in ("-", "null") else None

                if len(parts) > 1:
                    rest = parts[1].strip()
                    matches = re.findall(r"([^,;]+),\s*(\d+)", rest)
                    relevant_chunks = [
                        {"document_name": m[0].strip(), "page_number": int(m[1].strip())}
                        for m in matches
                    ]

            answer = None
            if raw_answer is not None:
                try:
                    if answer_type == "int":
                        answer = int(raw_answer)
                    elif answer_type == "float":
                        answer = float(raw_answer.replace(",", "."))
                    else:
                        answer = raw_answer
                except (ValueError, TypeError, AttributeError):
                    answer = None

            result = {
                "question_id": qid,
                "relevant_chunks": relevant_chunks,
                "answer": answer,
            }

            try:
                try:
                    with open(AUTOSAVE_PATH, "r", encoding="utf-8") as f:
                        all_results = json.load(f)
                except FileNotFoundError:
                    all_results = []

                all_results.append(result)

                with open(AUTOSAVE_PATH, "w", encoding="utf-8") as f:
                    json.dump(all_results, f, ensure_ascii=False, indent=4)

                logging.info(f"[Q{qid}] Saved result to {AUTOSAVE_PATH}")
            except Exception as e:
                logging.error(f"[Q{qid}] Failed to autosave: {e}")

            logging.info(f"[Q{qid}] Done")
            return result


# MAIN
async def main():
    logging.info("Starting pipeline")

    df = pd.read_excel(INPUT_PATH)
    logging.info(f"Loaded {len(df)} questions")

    pool = await asyncpg.create_pool(**DB_PARAMS, min_size=5, max_size=20)
    semaphore = asyncio.Semaphore(OPENAI_CONCURRENCY)

    try:
        tasks = [
            answer_question(
                pool=pool,
                qid=row["id"],
                question=row["full_question"],
                answer_type=row["answer_type"],
                semaphore=semaphore
            )
            for _, row in df.iterrows()
        ]
        results = await asyncio.gather(*tasks)

        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

        logging.info("Answers saved to answers.json")
    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
