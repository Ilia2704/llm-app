## 1) Install Python 3.12

**macOS (Homebrew)**

```bash
brew install python@3.12
echo 'export PATH="/opt/homebrew/opt/python@3.12/libexec/bin:$PATH"' >> ~/.zshrc && source ~/.zshrc
python3.12 --version
```

**Windows (PowerShell)**

```powershell
winget install -e --id Python.Python.3.12
python --version
```

**Ubuntu/Debian**

```bash
sudo add-apt-repository ppa:deadsnakes/ppa -y && sudo apt update
sudo apt install -y python3.12 python3.12-venv python3.12-dev
python3.12 --version
```

---

## 2) Create a virtual env & install deps

```bash
python3.12 -m venv llm-venv
source llm-venv/bin/activate        # Windows: .\llm-venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

**requirements.txt (pinned)**

```txt
ragas==0.3.4
pandas==2.3.2
python-dotenv==1.1.1
tqdm==4.67.1
rich==14.1.0
openai==1.107.3
langchain-openai==0.3.33
datasets==4.0.0
pyarrow==17.0.0
```

---

## 3) Add your OpenAI key

Create `.env` in the project root:

```dotenv
OPENAI_API_KEY=sk-YOUR_KEY_HERE
OPENAI_MODEL=gpt-4o
```

> If the file came from Windows, normalize line endings:
>
> ```bash
> sed -i '' 's/\r$//' .env
> ```

---

## 4) Run the auto-tests

```bash
# RAGAS evaluation (exits with code 1 if thresholds fail)
python run_ragas_demo_test.py

# or via pytest (if you added the sample test file)
pytest -q
```
