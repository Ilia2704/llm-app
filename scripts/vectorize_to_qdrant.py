
import argparse, re
from pathlib import Path
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance
from razdel import sentenize

from qdrant_io import test_connection, create_collection_if_needed, upsert_points

def read_text(path: Path, encoding: str = "cp1251") -> str:
    t = path.read_text(encoding=encoding, errors="ignore")

    # 1) приведение перевода строк и пробелов
    t = re.sub(r"\r\n?", "\n", t)
    t = re.sub(r"[ \t]+\n", "\n", t)

    # 2) склейка слов, разорванных переносом с дефисом:
    #    «гуманои- \n ды» -> «гуманоиды»
    t = re.sub(r"-\s*\n\s*", "", t)

    # 3) сжать избыточные пустые строки до максимум двух
    t = re.sub(r"\n{3,}", "\n\n", t)

    # 4) авто-«разворачивание» жёсткой верстки:
    #    одинарный перенос (не абзац) превращаем в пробел.
    #    (двойной перенос оставляем как абзац)
    #    пример: "Он сказал,\nчто..." -> "Он сказал, что..."
    #    но "...\n\nНовый абзац" — сохраняем.
    t = re.sub(r"(?<!\n)\n(?!\n)", " ", t)

    # 5) подчистка двойных пробелов и пробелов перед пунктуацией
    t = re.sub(r"[ \t]{2,}", " ", t)
    t = re.sub(r"\s+([,.;:!?…])", r"\1", t)

    return t.strip()


# --- 2) простая сегментация по «произведениям» (эвристика по заголовкам) ---
def split_works(txt: str) -> Dict[str, str]:
    lines = txt.splitlines()
    heads = []
    for i, l in enumerate(lines):
        s = l.strip()
        if 5 <= len(s) <= 80 and (s.isupper() or "«" in s or "»" in s):
            # окружён пустыми строками — вероятный заголовок
            if (i > 0 and not lines[i - 1].strip()) or (i + 1 < len(lines) and not lines[i + 1].strip()):
                heads.append(i)
    if not heads:
        return {"UNKNOWN": txt}
    heads = sorted(set([0] + heads + [len(lines)]))
    out = {}
    for a, b in zip(heads, heads[1:]):
        title = lines[a].strip() or f"Block_{a}"
        body = "\n".join(lines[a + 1 : b]).strip()
        if body:
            out[title[:60]] = body
    return out

# --- 3) разбиение на чанки по предложениям с overlap ---
def split_into_chunks(text: str, target_words: int = 300, overlap_sents: int = 2) -> List[str]:
    sents = [s.text.strip() for s in sentenize(text)]
    chunks, cur, n_words = [], [], 0
    for s in sents:
        w = len(s.split())
        if cur and n_words + w > target_words:
            chunks.append(" ".join(cur).strip())
            cur = cur[-overlap_sents:]
            n_words = sum(len(x.split()) for x in cur)
        cur.append(s)
        n_words += w
    if cur:
        chunks.append(" ".join(cur).strip())
    return chunks

# --- 4) векторизация ru-SBERT ---
def embed_ru_sbert(model: SentenceTransformer, texts: List[str]) -> List[List[float]]:
    # ru-SBERT не требует префиксов "passage:" / "query:"
    vecs = model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=True)
    return [v.tolist() for v in vecs]

# --- 5) конвейер: txt -> чанки -> эмбеддинги -> Qdrant ---
def run_pipeline(
    input_path: Path,
    encoding: str,
    host: str,
    port: int,
    collection: str,
    model_name: str,
    chunk_words: int,
    overlap_sents: int,
    batch_size: int,
):
    print(test_connection(host, port))
    txt = read_text(input_path, encoding=encoding)
    works = split_works(txt)

    model = SentenceTransformer(model_name)
    dim = model.get_sentence_embedding_dimension()

    client = QdrantClient(host=host, port=port)
    create_collection_if_needed(client, collection, dim, Distance.COSINE)

    total_points = 0
    for work_title, body in works.items():
        chunks = split_into_chunks(body, target_words=chunk_words, overlap_sents=overlap_sents)
        payloads = [{"work": work_title, "chunk_id": i, "text": ch} for i, ch in enumerate(chunks)]
        vectors = embed_ru_sbert(model, [p["text"] for p in payloads])
        total_points += upsert_points(client, collection, vectors, payloads, batch_size=batch_size)
        print(f"Indexed work: {work_title[:50]}… | chunks: {len(chunks)}")

    print(f"Done. Total points upserted: {total_points}")

# --- CLI ---
if __name__ == "__main__":
    ap = argparse.ArgumentParser("Vectorize TXT with ru-SBERT and push to Qdrant")
    ap.add_argument("--input", type=Path, default=Path("./data/strugatskyie.txt"))
    ap.add_argument("--encoding", type=str, default="cp1251")
    ap.add_argument("--host", type=str, default="localhost")
    ap.add_argument("--port", type=int, default=6333)
    ap.add_argument("--collection", type=str, default="strugatsky_kb")
    ap.add_argument("--model", type=str, default="ai-forever/sbert_large_nlu_ru")
    ap.add_argument("--chunk-words", type=int, default=300)
    ap.add_argument("--overlap-sents", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    run_pipeline(
        input_path=args.input,
        encoding=args.encoding,
        host=args.host,
        port=args.port,
        collection=args.collection,
        model_name=args.model,
        chunk_words=args.chunk_words,
        overlap_sents=args.overlap_sents,
        batch_size=args.batch_size,
    )
