"""
Evaluate RAG chunking configs with RAGAS metrics using Google Gemini.

Required environment variables:
  - GOOGLE_API_KEY

Inputs (rag_python/data by default):
  - Context-recall-eval-dataset.xlsx
  - all_collected_docs_cache_new.json  (or the PDF, if cache is missing)

Outputs (rag_python/outputs):
  - retrieval_results.json
  - ragas_evaluation_results.csv
  - chroma_store/<config_name>/
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

from rag_eval.compat import ensure_vertexai_shim
from rag_eval.paths import DATA_DIR, OUTPUT_DIR, ensure_dirs

ensure_vertexai_shim()

from datasets import Dataset  # noqa: E402
from ragas import evaluate  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.metrics import context_precision, context_recall, faithfulness  # noqa: E402

CHAT_MODEL = "gemini-3.5-flash"
EMBEDDING_MODEL = "gemini-embedding-2"

DEFAULT_PDF = DATA_DIR / "Business Statistics - A. Aczel, J. Sounderpandian.pdf"
DEFAULT_DATASET_XLSX = DATA_DIR / "Context-recall-eval-dataset.xlsx"
DEFAULT_DOCS_CACHE = DATA_DIR / "all_collected_docs_cache_new.json"
DEFAULT_RETRIEVAL_CACHE = OUTPUT_DIR / "retrieval_results.json"
DEFAULT_RESULTS_CSV = OUTPUT_DIR / "ragas_evaluation_results.csv"
DEFAULT_CHROMA_ROOT = OUTPUT_DIR / "chroma_store"

CHUNK_SIZES = [1000, 1200]
CHUNK_OVERLAPS = [200, 250]


def require_google_api_key() -> None:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    os.environ["GOOGLE_API_KEY"] = api_key


def load_eval_dataset(xlsx_path: Path) -> tuple[list[str], dict[str, str]]:
    if not xlsx_path.exists():
        raise FileNotFoundError(
            f"Eval dataset not found: {xlsx_path}\n"
            "Run: uv run rag-generate-dataset"
        )
    df = pd.read_excel(xlsx_path)
    if "user_input" not in df.columns or "reference" not in df.columns:
        raise ValueError(
            f"Dataset must contain 'user_input' and 'reference' columns. Found: {list(df.columns)}"
        )
    questions = df["user_input"].tolist()
    ground_truth_lookup = dict(zip(df["user_input"], df["reference"]))
    print(f"Loaded {len(questions)} evaluation questions from {xlsx_path}")
    return questions, ground_truth_lookup


def load_source_documents(docs_cache: Path, pdf_path: Path) -> list[Document]:
    if docs_cache.exists():
        print(f"Loading documents from cache: {docs_cache}")
        with docs_cache.open("r", encoding="utf-8") as f:
            cached_data = json.load(f)
        return [
            Document(page_content=item["page_content"], metadata=item.get("metadata", {}))
            for item in cached_data
        ]

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"Neither docs cache ({docs_cache}) nor PDF ({pdf_path}) was found."
        )

    print(f"Docs cache missing; loading PDF pages [48:100] from {pdf_path}")
    pages = PyPDFLoader(str(pdf_path)).load()
    docs = pages[48:100]
    serializable = [{"page_content": d.page_content, "metadata": d.metadata} for d in docs]
    docs_cache.parent.mkdir(parents=True, exist_ok=True)
    target = docs_cache.resolve() if docs_cache.is_symlink() else docs_cache
    with target.open("w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)
    print(f"Cached {len(docs)} pages to {target}")
    return docs


def build_qa_chain(llm: ChatGoogleGenerativeAI):
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an intelligent assistant who answers questions about a given document.",
            ),
            (
                "user",
                "Here is the document content: {context}. My question is: {question}",
            ),
        ]
    )
    return template | llm | StrOutputParser()


def config_names() -> list[str]:
    return [f"Recursive_{cs}_{co}" for cs in CHUNK_SIZES for co in CHUNK_OVERLAPS]


def get_or_create_vectordb(
    config_name: str,
    chunks: list[Document],
    embeddings: GoogleGenerativeAIEmbeddings,
    chroma_root: Path,
) -> Chroma:
    persist_path = chroma_root / config_name
    if persist_path.exists():
        print(f"Loading existing Chroma store: {persist_path}")
        return Chroma(
            persist_directory=str(persist_path),
            embedding_function=embeddings,
        )

    print(f"Creating Chroma store at {persist_path} with {len(chunks)} chunks")
    persist_path.mkdir(parents=True, exist_ok=True)
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist_path),
    )


def run_retrieval(
    questions: list[str],
    source_docs: list[Document],
    embeddings: GoogleGenerativeAIEmbeddings,
    chain,
    retrieval_cache: Path,
    chroma_root: Path,
    force_refresh: bool,
) -> dict:
    if retrieval_cache.exists() and not force_refresh:
        print(f"Loading retrieval results from cache: {retrieval_cache}")
        with retrieval_cache.open("r", encoding="utf-8") as f:
            return json.load(f)

    retrieval_results: dict = {}
    for current_chunk_size in CHUNK_SIZES:
        for current_chunk_overlap in CHUNK_OVERLAPS:
            config_name = f"Recursive_{current_chunk_size}_{current_chunk_overlap}"
            print(f"\n--- Processing config: {config_name} ---")

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=current_chunk_size,
                chunk_overlap=current_chunk_overlap,
                length_function=len,
                is_separator_regex=False,
            )
            chunks = splitter.split_documents(source_docs)
            print(f"Split into {len(chunks)} chunks")

            try:
                vectordb = get_or_create_vectordb(
                    config_name, chunks, embeddings, chroma_root
                )
                retriever = vectordb.as_retriever(search_kwargs={"k": 2})
                config_results = []

                for question in tqdm(questions, desc=f"Questions ({config_name})", leave=False):
                    try:
                        retrieved_docs = retriever.invoke(question)
                        doc_contents = [doc.page_content for doc in retrieved_docs]
                        combined_context = (
                            " ".join(doc_contents) if doc_contents else "No relevant context found."
                        )
                        answer_str = chain.invoke(
                            {"context": combined_context, "question": question}
                        )
                        config_results.append(
                            {
                                "question": question,
                                "answer": answer_str,
                                "contexts": doc_contents,
                            }
                        )
                    except Exception as exc:
                        print(f"Error processing question '{question}' for {config_name}: {exc}")
                        config_results.append(
                            {
                                "question": question,
                                "answer": "Error in retrieval",
                                "contexts": [],
                            }
                        )

                retrieval_results[config_name] = config_results
            except Exception as exc:
                print(f"ERROR building index for {config_name}: {exc}")
                retrieval_results[config_name] = []

    retrieval_cache.parent.mkdir(parents=True, exist_ok=True)
    with retrieval_cache.open("w", encoding="utf-8") as f:
        json.dump(retrieval_results, f, indent=2, ensure_ascii=False)
    print(f"\nCached retrieval results to {retrieval_cache}")
    return retrieval_results


def extract_aggregate_scores(ragas_result) -> dict[str, float]:
    """Normalize RAGAS evaluate() output into a flat metric dict."""
    if isinstance(ragas_result, dict):
        return {k: float(v) for k, v in ragas_result.items() if isinstance(v, (int, float))}

    if hasattr(ragas_result, "scores") and isinstance(ragas_result.scores, dict):
        return {
            k: float(v)
            for k, v in ragas_result.scores.items()
            if isinstance(v, (int, float))
        }

    try:
        as_dict = dict(ragas_result)
        return {k: float(v) for k, v in as_dict.items() if isinstance(v, (int, float))}
    except Exception:
        pass

    if hasattr(ragas_result, "to_pandas"):
        df = ragas_result.to_pandas()
        numeric = df.select_dtypes(include="number")
        return {col: float(numeric[col].mean()) for col in numeric.columns}

    raise TypeError(f"Unrecognized RAGAS result type: {type(ragas_result)}")


def run_ragas_evaluation(
    retrieval_results: dict,
    ground_truth_lookup: dict[str, str],
    llm: ChatGoogleGenerativeAI,
) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    ragas_llm = LangchainLLMWrapper(llm)

    for config_key in config_names():
        eval_data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

        if config_key not in retrieval_results:
            print(f"Warning: no retrieval results for '{config_key}', skipping.")
            continue

        for item in retrieval_results[config_key]:
            eval_data["question"].append(item["question"])
            eval_data["answer"].append(item["answer"])
            eval_data["contexts"].append(item["contexts"])
            gt = ground_truth_lookup.get(item["question"], "")
            eval_data["ground_truth"].append(
                " ".join(gt) if isinstance(gt, list) else str(gt or "")
            )

        if not eval_data["question"]:
            print(f"Warning: empty eval data for '{config_key}', skipping.")
            continue

        print(
            f"\n--- RAGAS evaluation for {config_key} ({len(eval_data['question'])} samples) ---"
        )
        ragas_dataset = Dataset.from_dict(eval_data)
        ragas_result = evaluate(
            dataset=ragas_dataset,
            metrics=[context_recall, context_precision, faithfulness],
            llm=ragas_llm,
            raise_exceptions=True,
        )
        scores = extract_aggregate_scores(ragas_result)
        summary[config_key] = scores
        print(f"Scores: {scores}")

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RAG with RAGAS + Gemini")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_XLSX)
    parser.add_argument("--docs-cache", type=Path, default=DEFAULT_DOCS_CACHE)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--retrieval-cache", type=Path, default=DEFAULT_RETRIEVAL_CACHE)
    parser.add_argument("--results-csv", type=Path, default=DEFAULT_RESULTS_CSV)
    parser.add_argument("--chroma-root", type=Path, default=DEFAULT_CHROMA_ROOT)
    parser.add_argument(
        "--force-refresh-retrieval",
        action="store_true",
        help="Ignore existing retrieval_results cache and re-run retrieval/generation",
    )
    parser.add_argument(
        "--skip-retrieval",
        action="store_true",
        help="Only run RAGAS scoring from an existing retrieval cache",
    )
    return parser.parse_args()


def main() -> None:
    ensure_dirs()
    args = parse_args()
    require_google_api_key()

    questions, ground_truth_lookup = load_eval_dataset(args.dataset)
    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0)
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    chain = build_qa_chain(llm)

    if args.skip_retrieval:
        if not args.retrieval_cache.exists():
            raise FileNotFoundError(
                f"--skip-retrieval set but cache missing: {args.retrieval_cache}"
            )
        with args.retrieval_cache.open("r", encoding="utf-8") as f:
            retrieval_results = json.load(f)
    else:
        source_docs = load_source_documents(args.docs_cache, args.pdf)
        retrieval_results = run_retrieval(
            questions=questions,
            source_docs=source_docs,
            embeddings=embeddings,
            chain=chain,
            retrieval_cache=args.retrieval_cache,
            chroma_root=args.chroma_root,
            force_refresh=args.force_refresh_retrieval,
        )

    summary = run_ragas_evaluation(retrieval_results, ground_truth_lookup, llm)
    if not summary:
        raise RuntimeError("No RAGAS scores were produced.")

    summary_df = pd.DataFrame.from_dict(summary, orient="index")
    print("\nAggregate RAGAS scores:")
    print(summary_df.to_string())

    args.results_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(args.results_csv, index=True)
    print(f"\nSaved results to {args.results_csv}")


if __name__ == "__main__":
    main()
