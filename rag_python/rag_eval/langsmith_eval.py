"""
LangSmith RAG evaluation using Google Gemini (chat + embeddings).

Required environment variables (in rag_python/.env or repo-root .env):
  - GOOGLE_API_KEY
  - LANGSMITH_API_KEY
"""

from __future__ import annotations

import os
from typing_extensions import Annotated, TypedDict

from langchain_community.document_loaders import WebBaseLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import Client, traceable

from rag_eval.paths import OUTPUT_DIR, ensure_dirs

os.environ["LANGSMITH_TRACING"] = "true"

CHAT_MODEL = "gemini-3.5-flash"
EMBEDDING_MODEL = "gemini-embedding-2"

URLS = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

EXAMPLES = [
    {
        "inputs": {"question": "How does the ReAct agent use self-reflection? "},
        "outputs": {
            "answer": (
                "ReAct integrates reasoning and acting, performing actions - such tools "
                "like Wikipedia search API - and then observing / reasoning about the tool outputs."
            )
        },
    },
    {
        "inputs": {
            "question": "What are the types of biases that can arise with few-shot prompting?"
        },
        "outputs": {
            "answer": (
                "The biases that can arise with few-shot prompting include "
                "(1) Majority label bias, (2) Recency bias, and (3) Common token bias."
            )
        },
    },
    {
        "inputs": {"question": "What are five types of adversarial attacks?"},
        "outputs": {
            "answer": (
                "Five types of adversarial attacks are (1) Token manipulation, "
                "(2) Gradient based attack, (3) Jailbreak prompting, "
                "(4) Human red-teaming, (5) Model red-teaming."
            )
        },
    },
]

DATASET_NAME = "Q&A"


class CorrectnessGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    correct: Annotated[bool, ..., "True if the answer is correct, False otherwise."]


class RelevanceGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    relevant: Annotated[
        bool, ..., "Provide the score on whether the answer addresses the question"
    ]


class GroundedGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    grounded: Annotated[
        bool, ..., "Provide the score on if the answer hallucinates from the documents"
    ]


class RetrievalRelevanceGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    relevant: Annotated[
        bool,
        ...,
        "True if the retrieved documents are relevant to the question, False otherwise",
    ]


CORRECTNESS_INSTRUCTIONS = """You are a teacher grading a quiz. You will be given a QUESTION, the GROUND TRUTH (correct) ANSWER, and the STUDENT ANSWER. Here is the grade criteria to follow:
(1) Grade the student answers based ONLY on their factual accuracy relative to the ground truth answer.
(2) Ensure that the student answer does not contain any conflicting statements.
(3) It is OK if the student answer contains more information than the ground truth answer, as long as it is factually accurate relative to the  ground truth answer.

Correctness:
A correctness value of True means that the student's answer meets all of the criteria.
A correctness value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""

RELEVANCE_INSTRUCTIONS = """You are a teacher grading a quiz. You will be given a QUESTION and a STUDENT ANSWER. Here is the grade criteria to follow:
(1) Ensure the STUDENT ANSWER is concise and relevant to the QUESTION
(2) Ensure the STUDENT ANSWER helps to answer the QUESTION

Relevance:
A relevance value of True means that the student's answer meets all of the criteria.
A relevance value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""

GROUNDED_INSTRUCTIONS = """You are a teacher grading a quiz. You will be given FACTS and a STUDENT ANSWER. Here is the grade criteria to follow:
(1) Ensure the STUDENT ANSWER is grounded in the FACTS.
(2) Ensure the STUDENT ANSWER does not contain "hallucinated" information outside the scope of the FACTS.

Grounded:
A grounded value of True means that the student's answer meets all of the criteria.
A grounded value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""

RETRIEVAL_RELEVANCE_INSTRUCTIONS = """You are a teacher grading a quiz. You will be given a QUESTION and a set of FACTS provided by the student. Here is the grade criteria to follow:
(1) You goal is to identify FACTS that are completely unrelated to the QUESTION
(2) If the facts contain ANY keywords or semantic meaning related to the question, consider them relevant
(3) It is OK if the facts have SOME information that is unrelated to the question as long as (2) is met

Relevance:
A relevance value of True means that the FACTS contain ANY keywords or semantic meaning related to the QUESTION and are therefore relevant.
A relevance value of False means that the FACTS are completely unrelated to the QUESTION.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""


def require_api_keys() -> None:
    google_key = os.getenv("GOOGLE_API_KEY")
    langsmith_key = os.getenv("LANGSMITH_API_KEY")
    if not google_key:
        raise ValueError(
            "GOOGLE_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    if not langsmith_key:
        raise ValueError(
            "LANGSMITH_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    os.environ["GOOGLE_API_KEY"] = google_key
    os.environ["LANGSMITH_API_KEY"] = langsmith_key


def build_rag_bot():
    """Load docs, build vector store, and return a traced RAG function."""
    docs = [WebBaseLoader(url).load() for url in URLS]
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0,
    )
    doc_splits = text_splitter.split_documents(docs_list)
    print(f"Total number of documents loaded: {len(docs_list)}")
    print(f"Total number of document chunks: {len(doc_splits)}")

    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = InMemoryVectorStore.from_documents(
        documents=doc_splits,
        embedding=embeddings,
    )
    retriever = vectorstore.as_retriever(k=6)

    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=1)

    @traceable()
    def rag_bot(question: str) -> dict:
        docs = retriever.invoke(question)
        docs_string = "".join(doc.page_content for doc in docs)
        instructions = f"""You are a helpful assistant who is good at analyzing source information and answering questions.
Use the following source documents to answer the user's questions.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.

Documents:
{docs_string}"""
        ai_msg = llm.invoke(
            [
                {"role": "system", "content": instructions},
                {"role": "user", "content": question},
            ]
        )
        return {"answer": ai_msg.content, "documents": docs}

    return rag_bot


def ensure_dataset(client: Client) -> None:
    if not client.has_dataset(dataset_name=DATASET_NAME):
        dataset = client.create_dataset(dataset_name=DATASET_NAME)
        client.create_examples(dataset_id=dataset.id, examples=EXAMPLES)
        print(f"Created LangSmith dataset: {DATASET_NAME}")
    else:
        print(f"Using existing LangSmith dataset: {DATASET_NAME}")


def build_evaluators():
    grader_llm = ChatGoogleGenerativeAI(
        model=CHAT_MODEL, temperature=0
    ).with_structured_output(CorrectnessGrade)

    relevance_llm = ChatGoogleGenerativeAI(
        model=CHAT_MODEL, temperature=0
    ).with_structured_output(RelevanceGrade)

    grounded_llm = ChatGoogleGenerativeAI(
        model=CHAT_MODEL, temperature=0
    ).with_structured_output(GroundedGrade)

    retrieval_relevance_llm = ChatGoogleGenerativeAI(
        model=CHAT_MODEL, temperature=0
    ).with_structured_output(RetrievalRelevanceGrade)

    def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
        answers = f"""\
QUESTION: {inputs['question']}
GROUND TRUTH ANSWER: {reference_outputs['answer']}
STUDENT ANSWER: {outputs['answer']}"""
        grade = grader_llm.invoke(
            [
                {"role": "system", "content": CORRECTNESS_INSTRUCTIONS},
                {"role": "user", "content": answers},
            ]
        )
        return grade["correct"]

    def relevance(inputs: dict, outputs: dict) -> bool:
        answer = f"QUESTION: {inputs['question']}\nSTUDENT ANSWER: {outputs['answer']}"
        grade = relevance_llm.invoke(
            [
                {"role": "system", "content": RELEVANCE_INSTRUCTIONS},
                {"role": "user", "content": answer},
            ]
        )
        return grade["relevant"]

    def groundedness(inputs: dict, outputs: dict) -> bool:
        doc_string = "\n\n".join(doc.page_content for doc in outputs["documents"])
        answer = f"FACTS: {doc_string}\nSTUDENT ANSWER: {outputs['answer']}"
        grade = grounded_llm.invoke(
            [
                {"role": "system", "content": GROUNDED_INSTRUCTIONS},
                {"role": "user", "content": answer},
            ]
        )
        return grade["grounded"]

    def retrieval_relevance(inputs: dict, outputs: dict) -> bool:
        doc_string = "\n\n".join(doc.page_content for doc in outputs["documents"])
        answer = f"FACTS: {doc_string}\nQUESTION: {inputs['question']}"
        grade = retrieval_relevance_llm.invoke(
            [
                {"role": "system", "content": RETRIEVAL_RELEVANCE_INSTRUCTIONS},
                {"role": "user", "content": answer},
            ]
        )
        return grade["relevant"]

    return [correctness, groundedness, relevance, retrieval_relevance]


def main() -> None:
    ensure_dirs()
    require_api_keys()

    rag_bot = build_rag_bot()
    client = Client()
    ensure_dataset(client)
    evaluators = build_evaluators()

    def target(inputs: dict) -> dict:
        return rag_bot(inputs["question"])

    experiment_results = client.evaluate(
        target,
        data=DATASET_NAME,
        evaluators=evaluators,
        experiment_prefix="rag-doc-relevance-gemini",
        metadata={"version": "LCEL context, gemini-2.0-flash"},
    )

    try:
        df = experiment_results.to_pandas()
        print("\nEvaluation results:")
        print(df.to_string())
        out_csv = OUTPUT_DIR / "langsmith_evaluation_results.csv"
        df.to_csv(out_csv, index=False)
        print(f"\nSaved local copy to {out_csv}")
    except Exception as exc:
        print(f"\nEvaluation finished. Could not convert results to pandas: {exc}")
        print(experiment_results)


if __name__ == "__main__":
    main()
