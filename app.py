#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, time
from functools import lru_cache

from fastapi import FastAPI
from pydantic import BaseModel
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

from prometheus_client import Counter, Gauge, Histogram, make_asgi_app
from prometheus_fastapi_instrumentator import Instrumentator

# ---------- ENV ----------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL     = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION     = os.getenv("QDRANT_COLLECTION", "strugatsky_kb")
MODEL_NAME     = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
APP_NAME       = os.getenv("APP_NAME", "llm-app")

# ---------- FastAPI ----------
app = FastAPI(title="RAG Inference Metrics Demo")

# ---------- ЕДИНЫЕ ЛЕЙБЛЫ ДЛЯ ВСЕХ МЕТРИК ----------
_LABELS = ["app", "exp", "qid"]

HIST_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32)

RAG_TTFB = Histogram("rag_ttfb_seconds", "Time to first token (TTFB)",
                     ["app","exp","qid"], buckets=HIST_BUCKETS)

RAG_INFERENCE_LATENCY = Histogram("rag_inference_latency_seconds", "Total LLM inference latency",
                                  ["app","exp","qid"], buckets=HIST_BUCKETS)


# ---------- Prometheus метрики (только с _LABELS) ----------
RAG_THROUGHPUT = Gauge("rag_inference_throughput_tokens_per_second", "Tokens generated per second", _LABELS)
RAG_CACHE_HIT = Counter("rag_cache_hits_total", "Cache hits", _LABELS)
RAG_CACHE_MISS = Counter("rag_cache_misses_total", "Cache misses", _LABELS)

# Базовые HTTP-метрики
Instrumentator().instrument(app).expose(app)   # /metrics
app.mount("/prometheus", make_asgi_app())      # альтернативный путь

# ---------- Core clients ----------
emb = SentenceTransformer("ai-forever/sbert_large_nlu_ru")
qdr = QdrantClient(url=QDRANT_URL)
client_llm = OpenAI(api_key=OPENAI_API_KEY)

# ---------- Schemas ----------
class Ask(BaseModel):
    query: str
    k: int = 6
    max_tokens: int = 400
    exp: str = "baseline"     # baseline | optimized
    qid: str = "q1"           # Идентификатор запроса для графиков

# ---------- Simple cache ----------
@lru_cache(maxsize=64)
def cached_query_vec(query: str):
    # только возвращаем вектор; метрики считаем в /ask
    return emb.encode(query, normalize_embeddings=True).tolist()

def build_prompt(query: str, ctx):
    sources = "\n\n".join([f"[{i+1}] ({c['work']}) {c['text']}" for i, c in enumerate(ctx)])
    sys_msg = ("Ты — литературный помощник. Отвечай по-русски, строго опираясь на источники ниже. "
               "Если в источниках нет ответа — так и скажи. В конце укажи номера [1], [2].")
    user_msg = f"Вопрос: {query}\n\nИсточники:\n{sources}"
    return sys_msg, user_msg

# ---------- Endpoint ----------
@app.post("/ask")
def ask(body: Ask):
    labels = {"app": APP_NAME, "exp": body.exp, "qid": body.qid}

    # 1) Retrieval: кэш on/off в зависимости от exp
    if body.exp == "optimized":
        try:
            qvec = cached_query_vec(body.query)
            RAG_CACHE_HIT.labels(**labels).inc()
        except Exception:
            qvec = emb.encode(body.query, normalize_embeddings=True).tolist()
            RAG_CACHE_MISS.labels(**labels).inc()
    else:
        # baseline: без кэша
        qvec = emb.encode(body.query, normalize_embeddings=True).tolist()
        RAG_CACHE_MISS.labels(**labels).inc()

    hits = qdr.search(COLLECTION, query_vector=qvec, limit=body.k, with_payload=True)
    ctx = [{"text": (h.payload or {}).get("text", ""),
            "work": (h.payload or {}).get("work", ""),
            "score": h.score} for h in hits]

    sys_msg, usr_msg = build_prompt(body.query, ctx)

    # 2) LLM streaming → TTFB / latency / throughput
    t0 = time.perf_counter()
    first_token_time = None
    token_count = 0
    out = []

    stream = client_llm.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "system", "content": sys_msg},
                  {"role": "user", "content": usr_msg}],
        stream=True,
        temperature=0.2,
        max_tokens=body.max_tokens,
    )

    for chunk in stream:
        if first_token_time is None:
            first_token_time = time.perf_counter()
            RAG_TTFB.labels(**labels).observe(first_token_time - t0)
        delta = chunk.choices[0].delta
        if delta and delta.content:
            out.append(delta.content)
            token_count += 1

    total = time.perf_counter() - t0
    RAG_INFERENCE_LATENCY.labels(**labels).observe(total)
    if token_count > 0:
        RAG_THROUGHPUT.labels(**labels).set(token_count / total)

    return {
        "answer": "".join(out).strip(),
        "ctx_used": ctx,
        "metrics": {
            "ttfb_s": round((first_token_time - t0), 3) if first_token_time else None,
            "inference_s": round(total, 3),
            "tokens": token_count,
        },
    }
