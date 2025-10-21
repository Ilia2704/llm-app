
"""
Функции:
- Читает txt (по умолчанию cp1251), чистит верстку
- Эвристически делит на "произведения" по заголовкам
- Делит на чанки по предложениям: ~300 слов, overlap 2 предложения
- Векторизует rusSBERT (ai-forever/sbert_large_nlu_ru, 1024, normalize=True)
- Пересоздает коллекцию в Qdrant и загружает чанки батчами
"""

import argparse
import re
import sys
import uuid
from pathlib import Path
from typing import List, Dict, Tuple

from razdel import sentenize
from sentence_transformers import SentenceTransformer

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    PointStruct,
)



def read_text(path: Path, encoding: str = "cp1251") -> str:
    t = path.read_text(encoding=encoding, errors="ignore")

    t = re.sub(r"\r\n?", "\n", t)
    t = re.sub(r"[ \t]+\n", "\n", t)

    t = re.sub(r"-\s*\n\s*", "", t)

    t = re.sub(r"\n{3,}", "\n\n", t)

    t = re.sub(r"(?<!\n)\n(?!\n)", " ", t)

    t = re.sub(r"[ \t]{2,}", " ", t)
    t = re.sub(r"\s+([,.;:!?…])", r"\1", t)

    return t.strip()


def split_works(txt: str) -> Dict[str, str]:
    """
    Эвристически делим на "произведения" по заголовкам (uppercase/«кавычки» + пустая строка вокруг).
    Если заголовков нет — вернём один блок UNKNOWN.
    """
    lines = txt.splitlines()
    heads = []
    for i, l in enumerate(lines):
        s = l.strip()
        if 5 <= len(s) <= 80 and (s.isupper() or "«" in s or "»" in s):
            if (i > 0 and not lines[i - 1].strip()) or (i + 1 < len(lines) and not lines[i + 1].strip()):
                heads.append(i)
    if not heads:
        return {"UNKNOWN": txt}

    heads = sorted(set([0] + heads + [len(lines)]))
    out = {}
    for a, b in zip(heads, heads[1:]):
        title = lines[a].strip() or f"Block_{a}"
        body = "\n".join(lines[a + 1:b]).strip()
        if body:
            out[title[:60]] = body
    return out


def split_into_chunks(text: str, target_words: int = 300, overlap_sents: int = 2) -> List[str]:
    """
    Длина — по целевому числу слов (порядка 300), overlap — кол-во предложений.
    """
    sents = [s.text.strip() for s in sentenize(text)]
    chunks, cur, n_words = [], [], 0
    for s in sents:
        w = len(s.split())
        if cur and n_words + w > target_words:
            chunks.append(" ".join(cur).strip())
            # overlap последними N предложениями
            cur = cur[-overlap_sents:] if overlap_sents > 0 else []
            n_words = sum(len(x.split()) for x in cur)
        cur.append(s)
        n_words += w
    if cur:
        chunks.append(" ".join(cur).strip())
    return chunks



def embed_ru_sbert(model: SentenceTransformer, texts: List[str]) -> List[List[float]]:
    """
    Русский SBERT: без префиксов, нормализация включена (для Cosine).
    """
    vecs = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=64,
        show_progress_bar=True,
    )
    return [v.tolist() for v in vecs]




def recreate_collection(client: QdrantClient, name: str, dim: int, distance=Distance.COSINE):
    """
    Полное пересоздание коллекции с 0.
    """
    print(f"Recreating collection '{name}' (dim={dim}, distance={distance}) ...")
    client.recreate_collection(
        collection_name=name,
        vectors_config=VectorParams(size=dim, distance=distance),
    )
    print("Collection recreated.")


def batched(iterable, n: int):
    buf = []
    for x in iterable:
        buf.append(x)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf


def upsert_batch(
    client: QdrantClient,
    collection: str,
    vectors: List[List[float]],
    payloads: List[Dict],
    ids: List[str],
):
    points = [
        PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i])
        for i in range(len(payloads))
    ]
    client.upsert(collection_name=collection, points=points)


def make_ids(work: str, n: int) -> List[str]:
    """
    Детерминированные UUIDv5 на основе (work, chunk_id), чтобы перезапуски не плодили дубликаты.
    """
    base_ns = uuid.uuid5(uuid.NAMESPACE_DNS, work)
    return [str(uuid.uuid5(base_ns, str(i))) for i in range(n)]




def run_pipeline(
    input_path: Path,
    encoding: str,
    url: str,
    api_key: str,
    collection: str,
    model_name: str,
    chunk_words: int,
    overlap_sents: int,
    batch_size: int,
    distance: str,
    recreate: bool,
) -> Tuple[int, int]:
    client = QdrantClient(url=url, api_key=api_key)

    print(f"Reading: {input_path} (encoding={encoding})")
    txt = read_text(input_path, encoding=encoding)
    works = split_works(txt)
    print(f"Detected works: {len(works)}")

    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    dim = model.get_sentence_embedding_dimension()
    print(f"Embedding dim: {dim}")

    dist_map = {
        "cosine": Distance.COSINE,
        "dot": Distance.DOT,
        "euclid": Distance.EUCLID,
    }
    dist = dist_map.get(distance.lower(), Distance.COSINE)
    if recreate:
        recreate_collection(client, collection, dim, dist)
    else:
        try:
            client.get_collection(collection)
        except Exception:
            recreate_collection(client, collection, dim, dist)

    total_chunks = 0
    total_points = 0
    for work_title, body in works.items():
        chunks = split_into_chunks(body, target_words=chunk_words, overlap_sents=overlap_sents)
        total_chunks += len(chunks)

        payloads = [
            {
                "work": work_title,
                "chunk_id": i,
                "text": ch,
            }
            for i, ch in enumerate(chunks)
        ]
        ids = make_ids(work_title, len(chunks))

        print(f"Indexing: {work_title[:50]}… | chunks={len(chunks)}")
        for batch_indices in batched(list(range(len(chunks))), batch_size):
            batch_texts = [payloads[i]["text"] for i in batch_indices]
            batch_vecs = embed_ru_sbert(model, batch_texts)
            upsert_batch(
                client=client,
                collection=collection,
                vectors=batch_vecs,
                payloads=[payloads[i] for i in batch_indices],
                ids=[ids[i] for i in batch_indices],
            )
            total_points += len(batch_indices)

    print(f"\nDone. Works: {len(works)} | Chunks: {total_chunks} | Points upserted: {total_points}")
    return total_chunks, total_points



def main():
    ap = argparse.ArgumentParser("Rebuild Qdrant collection from TXT with ru-SBERT and sentence-based chunking")
    ap.add_argument("--input", type=Path, default=Path("./data/strugatskyie.txt"))
    ap.add_argument("--encoding", type=str, default="cp1251")
    ap.add_argument("--url", type=str, default="http://localhost:6333")
    ap.add_argument("--api-key", type=str, default=None)
    ap.add_argument("--collection", type=str, default="strugatsky_kb")
    ap.add_argument("--model", type=str, default="ai-forever/sbert_large_nlu_ru")
    ap.add_argument("--chunk-words", type=int, default=300)
    ap.add_argument("--overlap-sents", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--distance", type=str, default="cosine", choices=["cosine", "dot", "euclid"])
    ap.add_argument("--no-recreate", action="store_true", help="Не пересоздавать коллекцию (создать при отсутствии)")
    args = ap.parse_args()

    try:
        run_pipeline(
            input_path=args.input,
            encoding=args.encoding,
            url=args.url,
            api_key=args.api_key,
            collection=args.collection,
            model_name=args.model,
            chunk_words=args.chunk_words,
            overlap_sents=args.overlap_sents,
            batch_size=args.batch_size,
            distance=args.distance,
            recreate=not args.no_recreate,  
        )
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"Failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
