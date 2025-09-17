import os, json, sys
from dataclasses import dataclass
from typing import List, Dict, Any
from dotenv import load_dotenv
from tqdm import tqdm
import pandas as pd

# OpenAI Responses API
from openai import OpenAI

# RAGAS
from ragas import evaluate
from ragas.metrics import Faithfulness, ResponseRelevancy, ContextPrecision, ContextRecall

# RAGAS ожидает LLM/эмбеддинги; удобный путь — враппер на LangChain OpenAI
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

load_dotenv()

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
client = OpenAI()

def extract_output_text(resp) -> str:
    """
    Универсальный вытаскиватель текста из Responses API.
    Поддерживает и удобный resp.output_text (если есть),
    и ручной разбор массива output[].content[].text.
    """
    # 1) удобное свойство (см. README SDK)
    if hasattr(resp, "output_text") and resp.output_text:
        return resp.output_text.strip()

    # 2) ручной разбор (cookbook пример показывает .output[0].content[0].text)
    try:
        parts = []
        for block in getattr(resp, "output", []):
            for c in getattr(block, "content", []):
                if getattr(c, "type", None) == "output_text" and getattr(c, "text", None):
                    parts.append(c.text)
        if parts:
            return "\n".join(parts).strip()
    except Exception:
        pass

    # 3) fallback — просто превратить в json
    return json.dumps(resp, ensure_ascii=False)

SYSTEM_RULES = (
    "Вы — ассистент службы поддержки компании. Отвечайте ТОЛЬКО фактами из контекста. "
    "Если ответа нет в контексте — скажите: «Не нашёл в предоставленном контексте»."
)

# Наша «база знаний»
DOCS = {
    "shipping": "Доставка: отправляем в тот же рабочий день. Бесплатно в Бразилии при заказе от 50 BRL. Поддержка: пн–пт, 09:00–18:00 BRT.",
    "returns":  "Возврат: в течение 30 дней для всех товаров; электроника — 15 дней.",
    "warranty": "Гарантия: 1 год на все товары; на батареи — 6 месяцев.",
    "stores":   "Магазины: Флорианополис и Сан-Паулу. Самовывоз доступен.",
    "noise":    "История бренда: мы любим футбол и мате. Этот текст не содержит правил доставки."
}

@dataclass
class Sample:
    question: str
    ground_truth: str
    contexts: List[str]

SAMPLES: List[Sample] = [
    # TC1: базовый факт по доставке (ожидаем high faithfulness/relevancy)
    Sample(
        question="Сколько стоит доставка по Бразилии и когда отправляете?",
        ground_truth="Бесплатно от 50 BRL. Отправляем в тот же рабочий день.",
        contexts=[DOCS["shipping"]]
    ),
    # TC2: возврат электроники (более жёсткое правило)
    Sample(
        question="Какой срок возврата для электроники?",
        ground_truth="15 дней для электроники.",
        contexts=[DOCS["returns"]]
    ),
    # TC3: шум в контексте → precision должен ухудшиться
    Sample(
        question="Где находятся физические магазины?",
        ground_truth="Флорианополис и Сан-Паулу.",
        contexts=[DOCS["stores"], DOCS["noise"], DOCS["noise"]]
    ),
    # TC4: проверка recall — даём много контекста, чтобы модель использовала нужный
    Sample(
        question="Какие часы работы службы поддержки?",
        ground_truth="Понедельник–пятница, 09:00–18:00 BRT.",
        contexts=[DOCS["shipping"], DOCS["returns"], DOCS["warranty"], DOCS["noise"]]
    ),
    # TC5: запрет на выдумки — ответа нет → модель должна признаться
    Sample(
        question="Есть ли магазин в Рио-де-Жанейро?",
        ground_truth="В предоставленном контексте нет информации о магазине в Рио.",
        contexts=[DOCS["stores"]]
    ),
    # TC6: смешанный вопрос (доставка+самовывоз) + шум
    Sample(
        question="Есть ли самовывоз и сколько стоит доставка по Бразилии?",
        ground_truth="Самовывоз доступен; доставка бесплатна от 50 BRL.",
        contexts=[DOCS["stores"], DOCS["shipping"], DOCS["noise"]]
    ),
]

def llm_answer(question: str, contexts: List[str]) -> str:
    joined_ctx = "\n---\n".join(contexts)
    prompt = f"{SYSTEM_RULES}\n\nКонтекст:\n{joined_ctx}\n\nВопрос: {question}\nКраткий ответ:"
    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=prompt,
        temperature=0,
    )
    return extract_output_text(resp)

def main():
    rows: List[Dict[str, Any]] = []
    print("\n[1/3] Генерация ответов моделью...")
    for s in tqdm(SAMPLES):
        answer = llm_answer(s.question, s.contexts)
        rows.append({
            "question": s.question,
            "answer": answer,
            "ground_truth": s.ground_truth,
            "contexts": s.contexts,
        })
    df = pd.DataFrame(rows)

    print("\n[2/3] Оценка метрик RAGAS...")
    # LLM/эмбеддинги для RAGAS (через LangChain-обёртку)
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model=OPENAI_MODEL, temperature=0))
    evaluator_embeddings = OpenAIEmbeddings()  # использует OpenAI client из переменных окружения

    result = evaluate(
        dataset=df,
        metrics=[Faithfulness(), ResponseRelevancy(), ContextPrecision(), ContextRecall()],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        show_progress=True,
    )

    print("\n[3/3] Итоги:")
    print(result)  # словарь метрик и их средних значений
    # Сохраним подробности по каждому примеру
    details = result.to_pandas()
    details.to_csv("ragas_result_detailed.csv", index=False)
    print("\nДетализированный отчёт: ragas_result_detailed.csv")

    # Пороговые значения (пример — подберите под свой домен)
    thresholds = {
        "faithfulness": 0.80,
        "response_relevancy": 0.70,
        "context_precision": 0.60,
        "context_recall": 0.70,
    }

    # провалим процесс, если не прошли пороги по среднему
    summary = result.to_dict()  # {'faithfulness': 0.9, ...}
    fails = {k: (summary.get(k, 1.0) < v) for k, v in thresholds.items()}
    failed = [k for k, bad in fails.items() if bad]
    if failed:
        print("\n[FAIL] Порог(и) не пройдены:", failed)
        sys.exit(1)
    else:
        print("\n[OK] Все пороги пройдены.")

if __name__ == "__main__":
    main()
    