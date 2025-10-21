from typing import List, Dict, Iterable
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
import uuid

def test_connection(host: str = "localhost", port: int = 6333) -> str:
    c = QdrantClient(host=host, port=port)
    # простой вызов, чтобы убедиться, что Qdrant жив и отвечает
    _ = c.get_collections()
    return f"Qdrant OK at {host}:{port}"

def create_collection_if_needed(
    client: QdrantClient,
    collection: str,
    dim: int,
    distance: Distance = Distance.COSINE,
):
    cols = [x.name for x in client.get_collections().collections]
    if collection not in cols:
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=dim, distance=distance),
        )

def batched(iterable: Iterable, n: int):
    buf = []
    for x in iterable:
        buf.append(x)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf

def upsert_points(
    client: QdrantClient,
    collection: str,
    vectors: List[List[float]],
    payloads: List[Dict],
    batch_size: int = 256,
):
    points = (
        PointStruct(id=str(uuid.uuid4()), vector=vec, payload=pl)
        for vec, pl in zip(vectors, payloads)
    )
    total = 0
    for chunk in batched(points, batch_size):
        client.upsert(collection_name=collection, points=chunk)
        total += len(chunk)
    return total
