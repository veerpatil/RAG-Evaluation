# RAG Gemini Python pipeline (uv)

## Setup

```bash
cd rag_python
cp .env.example .env   # then fill in GOOGLE_API_KEY (and LANGSMITH_API_KEY if needed)
uv sync                # uses Python 3.12 (.python-version)
```

Requires `uv` and Python 3.11–3.13 (pinned to 3.12 by default).

## Run

```bash
# 1) Generate synthetic eval dataset from the PDF
uv run rag-generate-dataset

# 2) Evaluate chunking configs with RAGAS
uv run rag-evaluate

# 3) LangSmith RAG evaluation (needs LANGSMITH_API_KEY)
uv run rag-langsmith
```

Or:

```bash
uv run python -m rag_eval.generate_dataset
uv run python -m rag_eval.evaluate
uv run python -m rag_eval.langsmith_eval
```

## Layout

```
rag_python/
├── pyproject.toml      # uv project + dependencies
├── .env.example
├── data/               # inputs (PDF / dataset / docs cache)
├── outputs/            # retrieval caches, CSV scores, Chroma, LangSmith export
└── rag_eval/           # Python package
    ├── generate_dataset.py
    ├── evaluate.py
    └── langsmith_eval.py
```
