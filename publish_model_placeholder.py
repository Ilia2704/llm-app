import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "model"


def main() -> None:
    model_name = (os.getenv("MODEL_NAME") or "").strip()
    if not model_name:
        raise RuntimeError("Нужно передать MODEL_NAME.")

    bucket = (os.getenv("LANGFUSE_S3_BUCKET") or "langfuse").strip()
    minio_user = (os.getenv("LANGFUSE_MINIO_ROOT_USER") or "minio").strip()
    minio_password = (os.getenv("LANGFUSE_MINIO_ROOT_PASSWORD") or "miniosecret").strip()
    docker_network = (os.getenv("MINIO_DOCKER_NETWORK") or "llm-app-observability_default").strip()
    object_prefix = (os.getenv("MODEL_PLACEHOLDER_PREFIX") or "passed-models").strip().strip("/")
    minio_endpoint = "http://langfuse-minio:9000"

    safe_name = _safe_filename(model_name)

    with tempfile.TemporaryDirectory(prefix="model_placeholder_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        placeholder = tmp_path / f"{safe_name}.txt"
        placeholder.write_text(model_name + "\n", encoding="utf-8")

        print(f"[UPLOAD] Модель прошла проверку: {model_name}", flush=True)
        print(
            f"[UPLOAD] Загружаю заглушку в MinIO: s3://{bucket}/{object_prefix}/{placeholder.name}",
            flush=True,
        )

        subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--network",
                docker_network,
                "--entrypoint",
                "/bin/sh",
                "-v",
                f"{tmp_path}:/artifacts",
                "minio/mc:latest",
                "-lc",
                (
                    f"mc alias set local {minio_endpoint} {minio_user} {minio_password} >/dev/null && "
                    f"mc mb --ignore-existing local/{bucket}/{object_prefix} >/dev/null && "
                    f"mc cp /artifacts/{placeholder.name} local/{bucket}/{object_prefix}/{placeholder.name}"
                ),
            ],
            check=True,
        )

        print(f"[UPLOAD] Готово: {placeholder.name}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[UPLOAD][FAIL] {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
