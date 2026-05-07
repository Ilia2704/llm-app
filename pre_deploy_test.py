import json
import os
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import mlflow
from dotenv import load_dotenv


warnings.filterwarnings(
    "ignore",
    message=r'Field "model_name" in PromptModelConfig has conflict with protected namespace "model_"\.',
    category=UserWarning,
)


load_dotenv()


OLLAMA_BASE_URL = (os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434/v1").strip()
OLLAMA_HEALTH_URL = OLLAMA_BASE_URL[:-3] + "/api/tags" if OLLAMA_BASE_URL.endswith("/v1") else OLLAMA_BASE_URL.rstrip("/") + "/api/tags"

MODEL_RUNS = [
    {
        "name": "ollama-qwen3-0.6b-q8",
        "provider": "ollama",
        "model": "hf.co/Qwen/Qwen3-0.6B-GGUF:Q8_0",
    },
    {
        "name": "ollama-qwen3-4b-q4",
        "provider": "ollama",
        "model": "hf.co/Qwen/Qwen3-4B-GGUF:Q4_K_M",
    },
    {
        "name": "ollama-qwen3-8b-q4",
        "provider": "ollama",
        "model": "hf.co/Qwen/Qwen3-8B-GGUF:Q4_K_M",
    },
    {
        "name": "yandexgpt-lite",
        "provider": "yandex",
        "model": (os.getenv("OPENAI_MODEL") or "").strip(),
    },
]


def get_model_runs() -> list[dict[str, str]]:
    selected_name = (os.getenv("PREDEPLOY_MODEL_NAME") or "").strip()
    if not selected_name:
        return MODEL_RUNS

    for model_cfg in MODEL_RUNS:
        if model_cfg["name"] == selected_name:
            return [model_cfg]

    raise RuntimeError(f"Неизвестная модель для pre-deploy теста: {selected_name}")


def _log_artifact_text(text: str, path: Path) -> None:
    path.write_text(text, encoding="utf-8")
    mlflow.log_artifact(str(path))


def _ollama_is_ready() -> bool:
    try:
        with urlopen(OLLAMA_HEALTH_URL, timeout=2) as response:
            return response.status == 200
    except (URLError, OSError):
        return False


def ensure_ollama_server() -> subprocess.Popen[str] | None:
    if _ollama_is_ready():
        return None

    process = subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )

    deadline = time.time() + 30
    while time.time() < deadline:
        if _ollama_is_ready():
            return process
        if process.poll() is not None:
            break
        time.sleep(1)

    raise RuntimeError("Не удалось запустить локальный сервер Ollama.")


def pull_ollama_model(model: str) -> None:
    subprocess.run(["ollama", "pull", model], check=True)


def stop_ollama_model(model: str) -> None:
    subprocess.run(["ollama", "stop", model], check=False)


def terminate_ollama_server(process: subprocess.Popen[str] | None) -> None:
    if process is None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def ensure_mlflow_experiment() -> None:
    base_name = (os.getenv("MLFLOW_EXPERIMENT_NAME") or "llm-app-demo").strip()
    experiment = mlflow.get_experiment_by_name(base_name)

    if experiment is None:
        mlflow.create_experiment(base_name, artifact_location="mlflow-artifacts:/")
        mlflow.set_experiment(base_name)
        return

    artifact_location = (experiment.artifact_location or "").strip()
    if artifact_location.startswith("mlflow-artifacts:"):
        mlflow.set_experiment(base_name)
        return

    fixed_name = f"{base_name}-artifacts"
    fixed_experiment = mlflow.get_experiment_by_name(fixed_name)
    if fixed_experiment is None:
        mlflow.create_experiment(fixed_name, artifact_location="mlflow-artifacts:/")
    mlflow.set_experiment(fixed_name)


def run_model_test(model_cfg: dict[str, str]) -> bool:
    env = os.environ.copy()
    env.update(
        {
            "TARGET_PROVIDER": model_cfg["provider"],
            "OPENAI_MODEL": model_cfg["model"],
            "EVAL_PROVIDER": "yandex",
            "EMBEDDING_PROVIDER": "yandex",
            "OLLAMA_BASE_URL": OLLAMA_BASE_URL,
            "OLLAMA_API_KEY": (os.getenv("OLLAMA_API_KEY") or "ollama").strip(),
        }
    )

    with tempfile.TemporaryDirectory(prefix="pre_deploy_test_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        result_json = tmp_path / "result.json"
        stdout_path = tmp_path / "stdout.txt"
        stderr_path = tmp_path / "stderr.txt"
        env["RESULT_JSON_PATH"] = str(result_json)
        stdout_text = ""
        stderr_text = ""
        payload = {
            "target_provider": model_cfg["provider"],
            "target_model": model_cfg["model"],
            "passed": False,
        }

        try:
            if model_cfg["provider"] == "ollama":
                pull_ollama_model(model_cfg["model"])

            completed = subprocess.run(
                [sys.executable, "run_ragas_demo_test.py"],
                env=env,
                capture_output=True,
                text=True,
            )
            stdout_text = completed.stdout
            stderr_text = completed.stderr
            payload["passed"] = completed.returncode == 0
            if result_json.exists():
                payload.update(json.loads(result_json.read_text(encoding="utf-8")))
        except Exception as exc:
            stderr_text = f"{type(exc).__name__}: {exc}\n"
            payload["error"] = str(exc)
        finally:
            if model_cfg["provider"] == "ollama":
                stop_ollama_model(model_cfg["model"])

        passed = bool(payload.get("passed", False))

        with mlflow.start_run(run_name=model_cfg["name"], nested=True):
            mlflow.log_params(
                {
                    "target_provider": model_cfg["provider"],
                    "target_model": model_cfg["model"],
                    "eval_provider": payload.get("eval_provider", ""),
                    "eval_model": payload.get("eval_model", ""),
                    "embedding_provider": payload.get("embedding_provider", ""),
                    "embedding_model": payload.get("embedding_model", ""),
                }
            )
            mlflow.set_tags(
                {
                    "test_type": "pre_deploy_ragas",
                    "status": "passed" if passed else "failed",
                    "deployment_ready": "true" if passed else "false",
                }
            )
            mlflow.log_metric("test_passed", 1.0 if passed else 0.0)
            for metric_name, metric_value in payload.get("summary", {}).items():
                try:
                    mlflow.log_metric(metric_name, float(metric_value))
                except Exception:
                    pass
            for case in payload.get("cases", []):
                case_id = int(case.get("case_id", 0))
                for metric_name, metric_value in (case.get("metrics") or {}).items():
                    if metric_value is None:
                        continue
                    try:
                        mlflow.log_metric(f"case_{case_id:02d}_{metric_name}", float(metric_value))
                    except Exception:
                        pass

            (tmp_path / "payload.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            mlflow.log_artifact(str(tmp_path / "payload.json"))
            _log_artifact_text(stdout_text, stdout_path)
            _log_artifact_text(stderr_text, stderr_path)

        print(f"\n=== {model_cfg['name']} ===")
        print(stdout_text)
        if stderr_text.strip():
            print(stderr_text, file=sys.stderr)
        if payload.get("summary"):
            print("Средние метрики модели:")
            for metric_name, metric_value in payload["summary"].items():
                print(f"  - {metric_name}: {float(metric_value):.4f}")
        if payload.get("cases"):
            print("Метрики по кейсам:")
            for case in payload["cases"]:
                metric_parts = []
                for metric_name, metric_value in (case.get("metrics") or {}).items():
                    metric_parts.append(
                        f"{metric_name}=" + ("N/A" if metric_value is None else f"{float(metric_value):.4f}")
                    )
                print(f"  - case_{int(case['case_id']):02d}: " + ", ".join(metric_parts))

        return passed


def main() -> None:
    model_runs = get_model_runs()

    for model_cfg in model_runs:
        if model_cfg["provider"] == "yandex" and not model_cfg["model"]:
            raise RuntimeError("Для модели Yandex нужно заполнить OPENAI_MODEL в .env.")

    if not MODEL_RUNS[-1]["model"] and any(m["provider"] == "yandex" for m in model_runs):
        raise RuntimeError("Для модели Yandex нужно заполнить OPENAI_MODEL в .env.")

    mlflow.set_tracking_uri((os.getenv("MLFLOW_TRACKING_URI") or "http://localhost:5001").strip())
    ensure_mlflow_experiment()

    ollama_process = ensure_ollama_server()
    failed: list[str] = []
    passed: list[str] = []

    try:
        with mlflow.start_run(run_name="pre_deploy_test"):
            mlflow.set_tags({"suite": "pre_deploy_test"})
            for model_cfg in model_runs:
                ok = run_model_test(model_cfg)
                if ok:
                    passed.append(model_cfg["name"])
                else:
                    failed.append(model_cfg["name"])

            mlflow.log_param("model_count", len(model_runs))
            mlflow.log_param("passed_models", ",".join(passed))
            mlflow.log_param("failed_models", ",".join(failed))
            mlflow.log_metric("passed_model_count", float(len(passed)))
            mlflow.log_metric("failed_model_count", float(len(failed)))
    finally:
        terminate_ollama_server(ollama_process)

    if failed:
        print("\n[FAIL] Не прошли модели:", ", ".join(failed))
        sys.exit(1)

    print("\n[OK] Все модели прошли pre-deploy тест.")


if __name__ == "__main__":
    main()
