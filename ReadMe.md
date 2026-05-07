# llm-app

Репозиторий pre-deploy проверки LLM-моделей.

Что здесь есть:
- `pre_deploy_test.py` — основной прогон 4 моделей:
  - `Qwen3-0.6B` через `Ollama`
  - `Qwen3-4B` через `Ollama`
  - `Qwen3-8B` через `Ollama`
  - `YandexGPT Lite`
- `run_ragas_demo_test.py` — одиночный прогон одной модели с метриками `RAGAS`
- `toxic_test.py` — отдельный judge-тест на токсичность и грубость
- `docker-compose.yaml` — локальные `MLflow` и `Langfuse`

## Что проверяется

Для каждого прогона `RAGAS` считаются:
- "faithfulness"
- "answer_relevancy"
- "context_precision"
- "context_recall"
- "qa_semantic_correctness"

Результаты пишутся:
- в `MLflow`
- в `Langfuse`

## Что должно быть готово

- активирован `.venv`
- заполнен `.env`
- запущен `Docker Desktop`
- установлен `Ollama`

## Запуск локальных сервисов

Поднять `MLflow` и `Langfuse`:

```bash
docker compose up -d
docker compose ps
```

URL:
- `MLflow`: `http://localhost:5001`
- `Langfuse`: `http://localhost:3000`

## Запуск тестов

Полный pre-deploy прогон:

```bash
./.venv/bin/python pre_deploy_test.py
```

Что делает скрипт:
- при необходимости поднимает `Ollama`
- по очереди скачивает локальную модель
- прогоняет тест
- выгружает модель из памяти
- пишет метрики и артефакты в `MLflow`

Одиночный `RAGAS`-прогон одной модели:

```bash
./.venv/bin/python run_ragas_demo_test.py
```

Отдельный тест токсичности:

```bash
./.venv/bin/python toxic_test.py
```

## Полезные команды

Остановить локально загруженные модели `Ollama`:

```bash
ollama ps | awk 'NR>1 {print $1}' | xargs -n1 ollama stop
```

Остановить локальные сервисы:

```bash
docker compose down
```

Полный сброс `Langfuse` и `MLflow`:

```bash
docker compose down -v
docker compose up -d
```

## CI

В `GitHub Actions` запускается:

```bash
python3 pre_deploy_test.py
python3 toxic_test.py
```

Workflow рассчитан на `self-hosted runner`.
