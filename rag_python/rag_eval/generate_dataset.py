"""
Generate a synthetic RAG evaluation dataset with RAGAS + Google Gemini.

Required environment variables:
  - GOOGLE_API_KEY

Outputs (under rag_python/ by default):
  - data/Context-recall-eval-dataset.xlsx
  - data/all_collected_docs_cache_new.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_eval.compat import ensure_vertexai_shim
from rag_eval.paths import DATA_DIR, ensure_dirs

ensure_vertexai_shim()

from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.testset import TestsetGenerator  # noqa: E402

CHAT_MODEL = "gemini-3.5-flash"
EMBEDDING_MODEL = "gemini-embedding-2"

DEFAULT_PDF = DATA_DIR / "Business Statistics - A. Aczel, J. Sounderpandian.pdf"
DEFAULT_DOCS_CACHE = DATA_DIR / "all_collected_docs_cache_new.json"
DEFAULT_DATASET_XLSX = DATA_DIR / "Context-recall-eval-dataset.xlsx"


def require_google_api_key() -> None:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    os.environ["GOOGLE_API_KEY"] = api_key


def load_pdf_pages(pdf_path: Path, start_page: int, end_page: int) -> list[Document]:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    pages = PyPDFLoader(str(pdf_path)).load()
    docs = pages[start_page:end_page]
    print(
        f"Loaded {len(pages)} PDF pages; using slice [{start_page}:{end_page}] -> {len(docs)} pages"
    )
    return docs


def split_documents(
    docs: list[Document], chunk_size: int, chunk_overlap: int
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})")
    return chunks


def cache_documents(docs: list[Document], cache_path: Path) -> None:
    serializable_docs = [
        {"page_content": doc.page_content, "metadata": doc.metadata} for doc in docs
    ]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a real file even if cache_path is currently a symlink to repo root
    target = cache_path
    if cache_path.is_symlink():
        target = cache_path.resolve()
    with target.open("w", encoding="utf-8") as f:
        json.dump(serializable_docs, f, ensure_ascii=False, indent=2)
    print(f"Documents cached to {target}")


def build_generator() -> TestsetGenerator:
    generator_llm = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0.3)
    )
    generator_embeddings = LangchainEmbeddingsWrapper(
        GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    )
    return TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate RAGAS test dataset with Gemini")
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF, help="Source PDF path")
    parser.add_argument("--start-page", type=int, default=48, help="Inclusive page index")
    parser.add_argument("--end-page", type=int, default=100, help="Exclusive page index")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    parser.add_argument("--testset-size", type=int, default=5)
    parser.add_argument("--docs-cache", type=Path, default=DEFAULT_DOCS_CACHE)
    parser.add_argument("--output-xlsx", type=Path, default=DEFAULT_DATASET_XLSX)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable RAGAS debugging logs during generation",
    )
    return parser.parse_args()


def main() -> None:
    ensure_dirs()
    args = parse_args()
    require_google_api_key()

    docs = load_pdf_pages(args.pdf, args.start_page, args.end_page)
    docs_for_ragas = split_documents(docs, args.chunk_size, args.chunk_overlap)
    cache_documents(docs_for_ragas, args.docs_cache)

    generator = build_generator()
    print(f"Generating testset (size={args.testset_size}) with {CHAT_MODEL}...")
    dataset = generator.generate_with_langchain_docs(
        docs_for_ragas,
        testset_size=args.testset_size,
        with_debugging_logs=args.debug,
    )

    df = dataset.to_pandas()
    print("\nGenerated dataset preview:")
    print(df.head().to_string())

    args.output_xlsx.parent.mkdir(parents=True, exist_ok=True)
    output_target = args.output_xlsx.resolve() if args.output_xlsx.is_symlink() else args.output_xlsx
    df.to_excel(output_target, index=False)
    print(f"\nSaved dataset to {output_target}")


if __name__ == "__main__":
    main()
