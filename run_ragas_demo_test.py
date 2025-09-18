"""
ДЕМО: Автотесты LLM-ответов с метриками RAGAS + fallback для answer_relevancy.

1) Генерирует ответы моделью OpenAI по маленькой «базе знаний» (имитация RAG без ретривера).
2) Считает метрики RAGAS (faithfulness, context_precision, context_recall).
3) Если включён fallback (по умолчанию), считает answer_relevancy как cos_sim(emb(Q), emb(A)) в [0..1].

ENV:
- OPENAI_API_KEY
- OPENAI_MODEL (default: gpt-4o-mini)
- RAGAS_EMBEDDING_MODEL (default: text-embedding-3-small)
- USE_SIMPLE_AR = "1" (default) — включить fallback answer_relevancy; "0" — попытаться RAGAS AnswerRelevancy
- THRESH_* — пороги метрик
"""

import os
import sys
import logging
from dataclasses import dataclass
from typing import List, Dict, Any

from dotenv import load_dotenv
from tqdm import tqdm
import pandas as pd

from openai import OpenAI

from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    ContextPrecision,
    ContextRecall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import OpenAIEmbeddings  # для остальных метрик
from langchain_openai import ChatOpenAI
from datasets import Dataset

load_dotenv()

OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY не найден. Добавьте его в .env или экспортируйте.")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
RAGAS_EMBEDDING_MODEL = os.getenv("RAGAS_EMBEDDING_MODEL", "text-embedding-3-small").strip()
USE_SIMPLE_AR = os.getenv("USE_SIMPLE_AR", "1").strip() not in ("0", "false", "False")

client = OpenAI()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("ragas_demo")

# ---------------- Данные для демо ----------------

SYSTEM_RULES = (
    "Вы — ассистент службы поддержки компании. Отвечайте ТОЛЬКО фактами из контекста. "
    "Если ответа нет в контексте — скажите: «Не нашёл в предоставленном контексте»."
)

DOCS = {
    "shipping": "Доставка: отправляем в тот же рабочий день. Бесплатно в России при заказе от 1000 RUB. Поддержка: пн–пт, 09:00–18:00.",
    "returns":  "Возврат: в течение 30 дней для всех товаров; электроника — 15 дней.",
    "warranty": "Гарантия: 1 год на все товары; на батареи — 6 месяцев.",
    "stores":   "Магазины: Москва и Санкт-Петербург. Самовывоз доступен.",
    "noise":    "История бренда: мы любим футбол и мате. Этот текст не содержит правил доставки."
}

@dataclass
class Sample:
    question: str
    ground_truth: str
    contexts: List[str]

SAMPLES: List[Sample] = [
    Sample(
        question="Сколько стоит доставка по России и когда отправляете?",
        ground_truth="Бесплатно от 1000 RUB. Отправляем в тот же рабочий день.",
        contexts=[DOCS["shipping"]],
    ),
    Sample(
        question="Какой срок возврата для электроники?",
        ground_truth="15 дней для электроники.",
        contexts=[DOCS["returns"]],
    ),
    Sample(
        question="Где находятся физические магазины?",
        ground_truth="Санкт-Петербург и Москва.",
        contexts=[DOCS["stores"], DOCS["noise"], DOCS["noise"]],
    ),
    Sample(
        question="Какие часы работы службы поддержки?",
        ground_truth="Понедельник–пятница, 09:00–18:00.",
        contexts=[DOCS["shipping"], DOCS["returns"], DOCS["warranty"], DOCS["noise"]],
    ),
    Sample(
        question="Есть ли магазин в Новороссийске?",
        ground_truth="В предоставленном контексте нет информации о магазине в Новороссийске.",
        contexts=[DOCS["stores"]],
    ),
    Sample(
        question="Есть ли самовывоз и сколько стоит доставка по России?",
        ground_truth="Самовывоз доступен; доставка бесплатна от 1000 RUB.",
        contexts=[DOCS["stores"], DOCS["shipping"], DOCS["noise"]],
    ),
    Sample(
        question="Какова столица Франции?",
        ground_truth="Париж.",
        contexts=[], # QA-режим (без контекста)
    ),
]

# Функции для LLM, эмбеддингов и метрик
def extract_output_text(resp) -> str:
    """Извлекает текст из Responses API (output_text или output[].content[].text)."""
    if hasattr(resp, "output_text") and resp.output_text:
        return resp.output_text.strip()
    try:
        parts = []
        for block in getattr(resp, "output", []):
            for c in getattr(block, "content", []):
                if getattr(c, "type", None) in ("output_text", "text") and getattr(c, "text", None):
                    parts.append(c.text)
        if parts:
            return "\n".join(parts).strip()
    except Exception:
        pass
    
    return str(resp)

def llm_answer(question: str, contexts: List[str]) -> str:
    """Генерация ответа через OpenAI Responses API (температура=0).
    Если contexts пуст — модель отвечает из своих знаний (QA-режим)."""
    if contexts and len(contexts) > 0:
        joined_ctx = "\n---\n".join(contexts)
        prompt = (
            f"{SYSTEM_RULES}\n\n"
            f"Контекст:\n{joined_ctx}\n\n"
            f"Вопрос: {question}\nКраткий ответ:"
        )
    else:
        # QA-режим: без контекста и без правила «только из контекста»
        prompt = (
            "Вы — фактологичный помощник. Отвечайте кратко и точно.\n\n"
            f"Вопрос: {question}\nКраткий ответ:"
        )

    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=prompt,
        temperature=0,
        store=False,
    )
    return extract_output_text(resp).strip()

def cosine(u: List[float], v: List[float]) -> float:
    """Косинусное сходство -> [0,1]."""
    import math
    s = sum(a*b for a, b in zip(u, v))
    nu = math.sqrt(sum(a*a for a in u))
    nv = math.sqrt(sum(b*b for b in v))
    if nu == 0.0 or nv == 0.0:
        return 0.0
    return max(0.0, min(1.0, (s / (nu * nv) + 1.0) / 2.0))

def embed_texts(texts: List[str], *, model: str, client: OpenAI) -> List[List[float]]:
    """Эмбеддинги напрямую через OpenAI Embeddings API"""
    res = client.embeddings.create(model=model, 
                                   input=texts)
    return [d.embedding for d in res.data]

def compute_simple_answer_relevancy_from_df(df_texts: pd.DataFrame, *, model: str, client: OpenAI) -> List[float]:
    """Surrogate для answer_relevancy: cos_sim(emb(Q), emb(A)) из ИСХОДНОГО df с колонками question/answer."""
    questions = df_texts["question"].astype(str).tolist()
    answers = df_texts["answer"].astype(str).tolist()
    q_vecs = embed_texts(questions, model=model, client=client)
    a_vecs = embed_texts(answers, model=model, client=client)
    return [cosine(q, a) for q, a in zip(q_vecs, a_vecs)]

def compute_answer_gt_similarity(df_texts: pd.DataFrame, *, model: str, client: OpenAI) -> List[float]:
    """Семантическая корректность ответа: cos_sim(emb(A), emb(GT)) в [0..1]."""
    answers = df_texts["answer"].astype(str).tolist()
    gts = df_texts["ground_truth"].astype(str).tolist()
    a_vecs = embed_texts(answers, model=model, client=client)
    g_vecs = embed_texts(gts, model=model, client=client)
    return [cosine(a, g) for a, g in zip(a_vecs, g_vecs)]


# Основной сценарий тестирования
def main() -> None:
    # Генерация ответов
    rows: List[Dict[str, Any]] = []
    print("\n[1/3] Генерация ответов моделью...")
    log.info("Начало генерации: %d кейсов, модель=%s", len(SAMPLES), OPENAI_MODEL)

    for s in tqdm(SAMPLES):
        answer = llm_answer(s.question, s.contexts)
        rows.append(
            {
                "question": s.question,
                "answer": answer,
                "ground_truth": s.ground_truth,
                "contexts": list(s.contexts),
            }
        )

    df = pd.DataFrame(rows)

    # Базовая валидация
    bad = []
    for i, r in df.iterrows():
        if not (isinstance(r["question"], str) and r["question"].strip()):
            bad.append((i, "question"))
        if not (isinstance(r["answer"], str) and r["answer"].strip()):
            bad.append((i, "answer"))
        if not (isinstance(r["ground_truth"], str) and r["ground_truth"].strip()):
            bad.append((i, "ground_truth"))
        # contexts может быть пустым списком в QA-режиме
        if not (isinstance(r["contexts"], list) and all(isinstance(c, str) for c in r["contexts"])):
            bad.append((i, "contexts"))


      # Оценка: RAGAS для кейсов с контекстом + универсальные QA-метрики
    print("\n[2/3] Оценка метрик...")
    evaluator_llm = LangchainLLMWrapper(
        ChatOpenAI(model=OPENAI_MODEL, temperature=0, api_key=OPENAI_API_KEY)
    )
    evaluator_embeddings = OpenAIEmbeddings(client=client, model=RAGAS_EMBEDDING_MODEL)

    rag_mask = df["contexts"].apply(lambda xs: isinstance(xs, list) and len(xs) > 0)
    details_all = pd.DataFrame(index=df.index)

    # 2.1 RAGAS (только там, где есть контекст)
    if rag_mask.any():
        hf_ds = Dataset.from_pandas(df.loc[rag_mask, ["question", "answer", "contexts", "ground_truth"]])
        metrics = [Faithfulness(), ContextPrecision(), ContextRecall()]
        if not USE_SIMPLE_AR:
            from ragas.metrics import AnswerRelevancy
            metrics.insert(1, AnswerRelevancy())

        log.info(
            "ragas.evaluate(): rows=%d, metrics=%s, emb_model=%s, USE_SIMPLE_AR=%s",
            len(hf_ds),
            [m.__class__.__name__ for m in metrics],
            RAGAS_EMBEDDING_MODEL,
            USE_SIMPLE_AR,
        )
        try:
            result = evaluate(
                dataset=hf_ds,
                metrics=metrics,
                llm=evaluator_llm,
                embeddings=evaluator_embeddings,
                show_progress=True,
                raise_exceptions=True,
                batch_size=4,
            )
            try:
                details_rag = result.to_pandas()
            except AttributeError:
                details_rag = pd.DataFrame(result)
            # синхронизируем индексы с исходным df
            details_rag.index = df.index[rag_mask]
            # вливаем только доступные колонки
            for col in details_rag.columns:
                details_all.loc[rag_mask, col] = details_rag[col]
        except Exception as e:
            print("\n[FAIL] RAGAS evaluate() завершился ошибкой:", type(e).__name__, str(e))
            sys.exit(1)

    #  AnswerRelevancy (fallback Q↔A) для всех строк
    ar_scores = compute_simple_answer_relevancy_from_df(df, model=RAGAS_EMBEDDING_MODEL, client=client)
    details_all["answer_relevancy"] = ar_scores

    # Семантическая корректность (A↔GT) для всех строк
    qa_sim = compute_answer_gt_similarity(df, model=RAGAS_EMBEDDING_MODEL, client=client)
    details_all["qa_semantic_correctness"] = qa_sim

    # Сводка и quality-gates
    print("\n[3/3] Итоги (средние значения метрик):")

    wanted = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "qa_semantic_correctness",   
    ]
    present = [c for c in wanted if c in details_all.columns]
    summary: Dict[str, float] = {c: float(details_all[c].mean(skipna=True)) for c in present}

    for k in wanted:
        v = summary.get(k, None)
        print(f"- {k:23}: " + ("N/A" if v is None or (v != v) else f"{v:.4f}"))

    log.info("Итоговые метрики: %s", summary)

    def env_float(name: str, default: float) -> float:
        try:
            return float(os.getenv(name, default))
        except Exception:
            return default

    thresholds = {
        "faithfulness": env_float("THRESH_FAITHFULNESS", 0.80),
        "answer_relevancy": env_float("THRESH_ANSWER_RELEVANCY", 0.70),
        "context_precision": env_float("THRESH_CONTEXT_PRECISION", 0.60),
        "context_recall": env_float("THRESH_CONTEXT_RECALL", 0.70),
        "qa_semantic_correctness": env_float("THRESH_QA_SIM", 0.80),  
    }

    # Подробный отчёт по кейсам
    metrics_df = None
    if "details_all" in locals():
        metrics_df = details_all.copy()
    else:
        metrics_df = pd.DataFrame(index=df.index)

    report = df.join(metrics_df, how="left")

    def _fmt(x):
        try:
            import math
            if x is None or (isinstance(x, float) and (math.isnan(x))):
                return "N/A"
            return f"{float(x):.4f}"
        except Exception:
            return "N/A"

    print("\n=== Подробный отчёт по кейсам ===")
    for i, r in report.iterrows():
        print(f"\n[{i+1}] Q: {r['question']}")
        print(f"     A: {r['answer']}")
        print(f"    GT: {r['ground_truth']}")
        parts = []
        for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "qa_semantic_correctness"]:
            if m in report.columns:
                parts.append(f"{m}={_fmt(r.get(m))}")
        print("Scores: " + (", ".join(parts) if parts else "нет доступных метрик"))


    failed = []
    for k, th in thresholds.items():
        if k not in present:
            continue  
        v = summary.get(k, None)
        if v is None or (v != v) or v < th:
            failed.append(k)

    if failed:
        print("\n[FAIL] Порог(и) не пройдены:", failed)
        sys.exit(1)
    else:
        print("\n[OK] Все пороги пройдены.")


if __name__ == "__main__":
    main()
