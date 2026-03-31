"""CLI script: RAGAS faithfulness + answer_relevancy eval for audio-RAG.

Usage:
    python eval/evaluate.py path/to/dataset.json

Dataset format (JSON array):
    [{"question": "...", "ground_truth": "..."}, ...]
"""
import argparse
import json
import os
import sys

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import AnswerRelevancy, Faithfulness

# Allow running from project root or from eval/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import EMBEDDING_MODEL, VLLM_BASE_URL, VLLM_MODEL
from pipeline.embed import get_chroma_collection, load_embedding_model
from pipeline.rag import query_rag


def build_ragas_llm() -> LangchainLLMWrapper:
    """Create a RAGAS LLM wrapper that calls the local vLLM server."""
    lc_llm = ChatOpenAI(model=VLLM_MODEL, base_url=VLLM_BASE_URL, api_key="dummy")
    return LangchainLLMWrapper(lc_llm)


def build_ragas_embeddings() -> LangchainEmbeddingsWrapper:
    """Create a RAGAS embeddings wrapper using the local HuggingFace model."""
    hf_emb = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return LangchainEmbeddingsWrapper(hf_emb)


def load_dataset(path: str) -> list[dict]:
    """Load and validate eval dataset from a JSON file."""
    with open(path) as f:
        data = json.load(f)
    required = {"question", "ground_truth"}
    for i, item in enumerate(data):
        missing = required - item.keys()
        if missing:
            raise ValueError(f"Item {i} is missing keys: {missing}")
    return data


def build_samples(data: list[dict]) -> list[SingleTurnSample]:
    """Run RAG on each question and wrap results as RAGAS SingleTurnSamples."""
    collection = get_chroma_collection()
    emb_model = load_embedding_model()
    samples = []
    for item in data:
        answer, chunks = query_rag(item["question"], collection, emb_model)
        samples.append(
            SingleTurnSample(
                user_input=item["question"],
                response=answer,
                retrieved_contexts=[c["text"] for c in chunks],
                reference=item["ground_truth"],
            )
        )
    return samples


def main() -> None:
    """Run RAGAS evaluation and print aggregated metric scores."""
    parser = argparse.ArgumentParser(description="RAGAS eval for audio-RAG")
    parser.add_argument("dataset", help="Path to JSON eval dataset")
    args = parser.parse_args()

    print(f"Loading dataset: {args.dataset}")
    data = load_dataset(args.dataset)

    print(f"Running RAG on {len(data)} questions…")
    samples = build_samples(data)

    ragas_llm = build_ragas_llm()
    ragas_emb = build_ragas_embeddings()
    metrics = [
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_emb),
    ]

    print("Evaluating with RAGAS…")
    result = evaluate(EvaluationDataset(samples=samples), metrics=metrics)

    print("\n=== RAGAS Results ===")
    df = result.to_pandas()
    for col in ["faithfulness", "answer_relevancy"]:
        if col in df.columns:
            print(f"  {col}: {df[col].mean():.4f}")
    print("\nPer-sample scores:")
    print(df[["user_input", "faithfulness", "answer_relevancy"]].to_string(index=False))


if __name__ == "__main__":
    main()
