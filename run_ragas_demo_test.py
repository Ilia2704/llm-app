"""
ДЕМО: Автотесты LLM-ответов с метриками RAGAS + fallback для answer_relevancy.

1) Генерирует ответы целевой моделью по маленькой «базе знаний».
2) Считает метрики RAGAS и fallback answer_relevancy.
3) По env может тестировать локальную модель через Ollama или YandexGPT.
"""

import json
import os
import sys
import logging
import warnings
from dataclasses import dataclass
from typing import List, Dict, Any

from dotenv import load_dotenv
from tqdm import tqdm
import pandas as pd

from openai import OpenAI
try:
    from langfuse.openai import OpenAI as LangfuseOpenAI
except Exception:
    LangfuseOpenAI = None

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

warnings.filterwarnings(
    "ignore",
    message=r"Importing .* from 'ragas\.metrics' is deprecated.*",
    category=DeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"LangchainLLMWrapper is deprecated.*",
    category=DeprecationWarning,
)

USE_SIMPLE_AR = os.getenv("USE_SIMPLE_AR", "1").strip() not in ("0", "false", "False")


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("ragas_demo")


def _prepare_langfuse_env() -> None:
    if not (os.getenv("LANGFUSE_HOST") or "").strip():
        host = (os.getenv("LANGFUSE_BASE_URL") or "").strip()
        if host:
            os.environ["LANGFUSE_HOST"] = host


def _get_openai_cls():
    _prepare_langfuse_env()
    if (
        LangfuseOpenAI is not None
        and (os.getenv("LANGFUSE_PUBLIC_KEY") or "").strip()
        and (os.getenv("LANGFUSE_SECRET_KEY") or "").strip()
    ):
        return LangfuseOpenAI
    return OpenAI


def _langfuse_enabled() -> bool:
    return (
        LangfuseOpenAI is not None
        and (os.getenv("LANGFUSE_PUBLIC_KEY") or "").strip()
        and (os.getenv("LANGFUSE_SECRET_KEY") or "").strip()
    )


def _langfuse_request_args(*, name: str) -> Dict[str, Any]:
    if not _langfuse_enabled():
        return {}
    return {
        "name": name,
    }


def _normalize_provider(name: str, default: str) -> str:
    return (os.getenv(name, default) or default).strip().lower()


def _get_yandex_auth() -> tuple[str, str]:
    yc_api_key = (os.getenv("YC_API_KEY") or "").strip()
    yc_folder_id = (os.getenv("YC_FOLDER_ID") or "").strip()
    if not yc_api_key or not yc_folder_id:
        raise RuntimeError("Нужны YC_API_KEY и YC_FOLDER_ID в .env")
    return yc_api_key, yc_folder_id


def _default_chat_model(provider: str) -> str:
    if provider in ("yandex", "yandexgpt"):
        _, folder_id = _get_yandex_auth()
        return f"gpt://{folder_id}/yandexgpt-lite/latest"
    if provider == "ollama":
        return "hf.co/Qwen/Qwen3-4B-GGUF:Q4_K_M"
    raise RuntimeError(f"Неизвестный provider: {provider}")


def _default_embedding_model(provider: str) -> str:
    if provider in ("yandex", "yandexgpt"):
        _, folder_id = _get_yandex_auth()
        return f"emb://{folder_id}/text-embeddings/latest"
    if provider == "ollama":
        return "nomic-embed-text"
    raise RuntimeError(f"Неизвестный embedding provider: {provider}")


def _evaluator_observation_name(eval_model: str) -> str:
    target_model = (os.getenv("OPENAI_MODEL") or "").strip() or "unknown-target"
    return f"evaluator::{target_model}::via::{eval_model}"


def _make_openai_client(provider: str):
    client_cls = _get_openai_cls()

    if provider == "ollama":
        return client_cls(
            api_key=(os.getenv("OLLAMA_API_KEY") or "ollama").strip(),
            base_url=(os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434/v1").strip(),
        )

    if provider in ("yandex", "yandexgpt"):
        yc_api_key, yc_folder_id = _get_yandex_auth()
        return client_cls(
            api_key="DUMMY",
            base_url="https://llm.api.cloud.yandex.net/v1",
            default_headers={
                "Authorization": f"Api-Key {yc_api_key}",
                "OpenAI-Project": yc_folder_id,
            },
        )

    raise RuntimeError(f"Неизвестный provider: {provider}")


def _make_chat_model(provider: str, model: str):
    if provider == "ollama":
        return ChatOpenAI(
            model=model,
            temperature=0,
            model_kwargs={"name": _evaluator_observation_name(model)},
            api_key=(os.getenv("OLLAMA_API_KEY") or "ollama").strip(),
            base_url=(os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434/v1").strip(),
        )

    if provider in ("yandex", "yandexgpt"):
        yc_api_key, yc_folder_id = _get_yandex_auth()
        return ChatOpenAI(
            model=model,
            temperature=0,
            model_kwargs={"name": _evaluator_observation_name(model)},
            api_key="DUMMY",
            base_url="https://llm.api.cloud.yandex.net/v1",
            default_headers={
                "Authorization": f"Api-Key {yc_api_key}",
                "OpenAI-Project": yc_folder_id,
            },
        )

    raise RuntimeError(f"Неизвестный evaluator provider: {provider}")

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
    "noise":    "История бренда: мы любим футбол и пиво. Этот текст не содержит правил доставки."
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

# Генерация ответа
def llm_answer(question: str, contexts: List[str], *, client: OpenAI, model: str, case_id: int) -> str:
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

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_RULES} if contexts else
            {"role": "system", "content": "Вы — фактологичный помощник. Отвечайте кратко и точно."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=300,
        **_langfuse_request_args(
            name=f"{model}::chat::case_{case_id:02d}",
        ),
    )

    return (resp.choices[0].message.content or "").strip()


# Косинусное сходство
def cosine(u: List[float], v: List[float]) -> float:
    """Косинусное сходство -> [0,1]."""
    import math
    s = sum(a*b for a, b in zip(u, v))
    nu = math.sqrt(sum(a*a for a in u))
    nv = math.sqrt(sum(b*b for b in v))
    if nu == 0.0 or nv == 0.0:
        return 0.0
    return max(0.0, min(1.0, (s / (nu * nv) + 1.0) / 2.0))

# Эмбеддинги
def embed_one(text: str, *, model: str, client: OpenAI, name: str) -> List[float]:
    res = client.embeddings.create(
        model=model,
        input=text,
        encoding_format="float",
        **_langfuse_request_args(name=name),
    )
    return res.data[0].embedding

# Метрики
def compute_simple_answer_relevancy_from_df(
    df_texts: pd.DataFrame, *, model: str, client: OpenAI, target_model: str
) -> List[float]:
    """Surrogate для answer_relevancy: cos_sim(emb(Q), emb(A)) из ИСХОДНОГО df с колонками question/answer."""
    scores: List[float] = []
    for idx, row in df_texts.iterrows():
        question = str(row["question"]).strip()
        answer = str(row["answer"]).strip()
        if not answer:
            scores.append(0.0)
            continue
        q_vec = embed_one(
            question,
            model=model,
            client=client,
            name=f"{target_model}::embedding::question::case_{idx + 1:02d}",
        )
        a_vec = embed_one(
            answer,
            model=model,
            client=client,
            name=f"{target_model}::embedding::answer::case_{idx + 1:02d}",
        )
        scores.append(cosine(q_vec, a_vec))
    return scores

# Метрики
def compute_answer_gt_similarity(
    df_texts: pd.DataFrame, *, model: str, client: OpenAI, target_model: str
) -> List[float]:
    """Семантическая корректность ответа: cos_sim(emb(A), emb(GT)) в [0..1]."""
    scores: List[float] = []
    for idx, row in df_texts.iterrows():
        answer = str(row["answer"]).strip()
        ground_truth = str(row["ground_truth"]).strip()
        if not answer:
            scores.append(0.0)
            continue
        a_vec = embed_one(
            answer,
            model=model,
            client=client,
            name=f"{target_model}::embedding::answer_gt::case_{idx + 1:02d}",
        )
        gt_vec = embed_one(
            ground_truth,
            model=model,
            client=client,
            name=f"{target_model}::embedding::ground_truth::case_{idx + 1:02d}",
        )
        scores.append(cosine(a_vec, gt_vec))
    return scores


# Основной сценарий тестирования
def main() -> None:
    target_provider = _normalize_provider("TARGET_PROVIDER", "yandex")
    eval_provider = _normalize_provider("EVAL_PROVIDER", "yandex")
    embedding_provider = _normalize_provider("EMBEDDING_PROVIDER", "yandex")

    target_model = (os.getenv("OPENAI_MODEL") or _default_chat_model(target_provider)).strip()
    eval_model = (os.getenv("EVAL_MODEL") or _default_chat_model(eval_provider)).strip()
    ragas_embedding_model = (
        os.getenv("RAGAS_EMBEDDING_MODEL") or _default_embedding_model(embedding_provider)
    ).strip()

    target_client = _make_openai_client(target_provider)
    embedding_client = _make_openai_client(embedding_provider)

    # Генерация ответов
    rows: List[Dict[str, Any]] = []
    print("\n[1/3] Генерация ответов моделью...")
    log.info(
        "Начало генерации: кейсов=%d, target_provider=%s, target_model=%s, eval_provider=%s, eval_model=%s, embedding_provider=%s, embedding_model=%s",
        len(SAMPLES),
        target_provider,
        target_model,
        eval_provider,
        eval_model,
        embedding_provider,
        ragas_embedding_model,
    )

    for case_id, s in enumerate(tqdm(SAMPLES), start=1):
        answer = llm_answer(s.question, s.contexts, client=target_client, model=target_model, case_id=case_id)
        rows.append(
            {
                "case_id": case_id,
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
        _make_chat_model(eval_provider, eval_model)
    )

    evaluator_embeddings = OpenAIEmbeddings(client=embedding_client, model=ragas_embedding_model)

    rag_mask = df["contexts"].apply(lambda xs: isinstance(xs, list) and len(xs) > 0)
    details_all = pd.DataFrame(index=df.index)


    # RAGAS (только там, где есть контекст)
   
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
            ragas_embedding_model,
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

    # AnswerRelevancy (fallback Q↔A) для всех строк
    ar_scores = compute_simple_answer_relevancy_from_df(
        df,
        model=ragas_embedding_model,
        client=embedding_client,
        target_model=target_model,
    )
    details_all["answer_relevancy"] = ar_scores

    # Семантическая корректность (A↔GT) для всех строк
    qa_sim = compute_answer_gt_similarity(
        df,
        model=ragas_embedding_model,
        client=embedding_client,
        target_model=target_model,
    )
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
        "answer_relevancy": env_float("THRESH_ANSWER_RELEVANCY", 0.50),
        "context_precision": env_float("THRESH_CONTEXT_PRECISION", 0.50),
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
    case_results = []
    for i, r in report.iterrows():
        case_id = int(r.get("case_id", i + 1))
        print(f"\n[{case_id}] Q: {r['question']}")
        print(f"     A: {r['answer']}")
        print(f"    GT: {r['ground_truth']}")
        parts = []
        metrics_for_case: Dict[str, Any] = {}
        for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "qa_semantic_correctness"]:
            if m in report.columns:
                raw_value = r.get(m)
                parts.append(f"{m}={_fmt(raw_value)}")
                try:
                    import math
                    if raw_value is None or (isinstance(raw_value, float) and math.isnan(raw_value)):
                        metrics_for_case[m] = None
                    else:
                        metrics_for_case[m] = float(raw_value)
                except Exception:
                    metrics_for_case[m] = None
        print("Scores: " + (", ".join(parts) if parts else "нет доступных метрик"))
        case_results.append(
            {
                "case_id": case_id,
                "question": r["question"],
                "answer": r["answer"],
                "ground_truth": r["ground_truth"],
                "metrics": metrics_for_case,
            }
        )


    failed = []
    for k, th in thresholds.items():
        if k not in present:
            continue  
        v = summary.get(k, None)
        if v is None or (v != v) or v < th:
            failed.append(k)

    result_payload = {
        "target_provider": target_provider,
        "target_model": target_model,
        "eval_provider": eval_provider,
        "eval_model": eval_model,
        "embedding_provider": embedding_provider,
        "embedding_model": ragas_embedding_model,
        "summary": summary,
        "cases": case_results,
        "thresholds": thresholds,
        "failed_metrics": failed,
        "passed": not failed,
    }

    result_json_path = (os.getenv("RESULT_JSON_PATH") or "").strip()
    if result_json_path:
        with open(result_json_path, "w", encoding="utf-8") as f:
            json.dump(result_payload, f, ensure_ascii=False, indent=2)

    if failed:
        print("\n[FAIL] Порог(и) не пройдены:", failed)
        sys.exit(1)
    else:
        print("\n[OK] Все пороги пройдены.")


if __name__ == "__main__":
    main()
