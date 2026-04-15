"""DeepEval evaluation for Echo RAG.

QA pairs are synthesized from indexed chunks using GPT-4o-mini,
then the RAG pipeline answers them (using local vLLM), and GPT-4o-mini
judges the results on four metrics:
  - Faithfulness         (answer grounded in retrieved context?)
  - Answer Relevancy     (answer addresses the question?)
  - Contextual Precision (top chunks are the most relevant?)
  - Contextual Recall    (chunks contain what's needed to answer?)

Usage:
    # Generate QA pairs, run eval, print results
    uv run python eval/evaluate.py --user-id <session-uuid>

    # Also save generated QA pairs for inspection / re-use
    uv run python eval/evaluate.py --user-id <uuid> --save eval/qa_pairs.json

    # Skip synthesis, re-use saved QA pairs
    uv run python eval/evaluate.py --user-id <uuid> --dataset eval/qa_pairs.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepeval import evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
)
from deepeval.models import GPTModel
from deepeval.synthesizer import Synthesizer
from deepeval.test_case import LLMTestCase

from config import OPENAI_API_KEY  # noqa: F401 — triggers load_dotenv()
from pipeline.embed import get_chroma_collection, load_embedding_model
from pipeline.rag import generate_answer, retrieve

JUDGE_MODEL = "gpt-4o-mini"


def fetch_chunks(user_id: str) -> list[str]:
    """Fetch all RAG chunk documents from ChromaDB, excluding metadata-only docs."""
    collection = get_chroma_collection(user_id)
    result = collection.get(include=["documents", "metadatas"])
    return [
        doc
        for doc, meta in zip(result["documents"], result["metadatas"])
        if meta.get("type") not in ("summary", "show_notes")
    ]


def synthesize(chunks: list[str], n: int) -> list[dict]:
    """Generate n QA pairs from chunks using GPT-4o-mini via DeepEval Synthesizer."""
    judge = GPTModel(model=JUDGE_MODEL)
    synthesizer = Synthesizer(model=judge)
    goldens = synthesizer.generate_goldens_from_contexts(
        contexts=[[c] for c in chunks[:n]],
    )
    return [
        {"question": g.input, "expected_answer": g.expected_output}
        for g in goldens
    ]


def build_test_cases(qa_pairs: list[dict], user_id: str) -> list[LLMTestCase]:
    """Run each question through the RAG pipeline and wrap as DeepEval test cases."""
    collection = get_chroma_collection(user_id)
    emb_model = load_embedding_model()
    test_cases = []
    for i, pair in enumerate(qa_pairs, 1):
        print(f"  [{i}/{len(qa_pairs)}] {pair['question'][:80]}")
        chunks = retrieve(pair["question"], collection, emb_model)
        answer = generate_answer(pair["question"], chunks)
        test_cases.append(
            LLMTestCase(
                input=pair["question"],
                actual_output=answer,
                expected_output=pair.get("expected_answer", ""),
                retrieval_context=[c["text"] for c in chunks],
            )
        )
    return test_cases


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepEval for Echo RAG")
    parser.add_argument("--user-id", required=True, help="Session UUID (shown in app sidebar)")
    parser.add_argument("--dataset", help="Path to existing QA pairs JSON (skips synthesis)")
    parser.add_argument("--save", help="Path to save generated QA pairs JSON")
    parser.add_argument("--n", type=int, default=20, help="QA pairs to generate (default: 20)")
    args = parser.parse_args()

    if args.dataset:
        print(f"Loading QA pairs from {args.dataset}")
        with open(args.dataset) as f:
            qa_pairs = json.load(f)
    else:
        print("Fetching chunks from ChromaDB…")
        chunks = fetch_chunks(args.user_id)
        print(f"Found {len(chunks)} chunks. Synthesizing {args.n} QA pairs with {JUDGE_MODEL}…")
        qa_pairs = synthesize(chunks, args.n)
        print(f"Generated {len(qa_pairs)} QA pairs.")

    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        with open(args.save, "w") as f:
            json.dump(qa_pairs, f, indent=2, ensure_ascii=False)
        print(f"QA pairs saved to {args.save}")

    print(f"\nRunning RAG on {len(qa_pairs)} questions…")
    test_cases = build_test_cases(qa_pairs, args.user_id)

    judge = GPTModel(model=JUDGE_MODEL)
    metrics = [
        FaithfulnessMetric(model=judge, threshold=0.7),
        AnswerRelevancyMetric(model=judge, threshold=0.7),
        ContextualPrecisionMetric(model=judge, threshold=0.7),
        ContextualRecallMetric(model=judge, threshold=0.7),
    ]

    print(f"\nEvaluating with {JUDGE_MODEL}…")
    results = evaluate(test_cases, metrics, print_results=False)

    print("\n=== Results ===")
    scores: dict[str, list[float]] = {}
    for tc in results.test_results:
        for m in tc.metrics_data:
            scores.setdefault(m.name, []).append(m.score)

    for name, vals in scores.items():
        avg = sum(vals) / len(vals)
        print(f"  {name}: {avg:.3f}  (n={len(vals)})")


if __name__ == "__main__":
    main()
