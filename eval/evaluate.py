"""DeepEval evaluation for Echo RAG.

Generates QA pairs from indexed chunks using DeepEval Synthesizer,
runs them through the RAG pipeline, and scores with:
  - Faithfulness       (is the answer grounded in the retrieved context?)
  - Answer Relevancy   (does the answer address the question?)
  - Contextual Precision (are the top chunks the most relevant ones?)
  - Contextual Recall    (do the chunks contain what's needed to answer?)

Usage:
    # Generate QA pairs, run eval, print results
    uv run python eval/evaluate.py --user-id <your-session-uuid>

    # Save generated QA pairs for inspection
    uv run python eval/evaluate.py --user-id <uuid> --save eval/qa_pairs.json

    # Re-use saved QA pairs (skip synthesis)
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
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.synthesizer import Synthesizer
from deepeval.test_case import LLMTestCase
from openai import OpenAI

from config import VLLM_BASE_URL, VLLM_MODEL
from pipeline.embed import get_chroma_collection, load_embedding_model
from pipeline.rag import generate_answer, retrieve


class LocalVLLM(DeepEvalBaseLLM):
    """DeepEval-compatible wrapper around the local vLLM server."""

    def __init__(self) -> None:
        self._client = OpenAI(base_url=VLLM_BASE_URL, api_key="dummy")

    def load_model(self):
        return self._client

    def generate(self, prompt: str, schema=None) -> str:
        response = self._client.chat.completions.create(
            model=VLLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.0,
        )
        return response.choices[0].message.content

    async def a_generate(self, prompt: str, schema=None) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return VLLM_MODEL


def fetch_chunks(user_id: str) -> list[str]:
    """Pull all RAG chunks from ChromaDB, excluding metadata-only docs."""
    collection = get_chroma_collection(user_id)
    result = collection.get(include=["documents", "metadatas"])
    return [
        doc
        for doc, meta in zip(result["documents"], result["metadatas"])
        if meta.get("type") not in ("summary", "show_notes")
    ]


def synthesize(chunks: list[str], model: LocalVLLM, n: int) -> list[dict]:
    """Generate n QA pairs from chunks using DeepEval Synthesizer."""
    synthesizer = Synthesizer(model=model)
    # Each golden gets its own single-chunk context
    goldens = synthesizer.generate_goldens_from_contexts(
        contexts=[[c] for c in chunks[:n]],
    )
    return [
        {"question": g.input, "expected_answer": g.expected_output}
        for g in goldens
    ]


def build_test_cases(
    qa_pairs: list[dict],
    user_id: str,
) -> list[LLMTestCase]:
    """Run RAG on each question and wrap as DeepEval test cases."""
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
    parser.add_argument("--user-id", required=True, help="Session UUID (shown in sidebar)")
    parser.add_argument("--dataset", help="Path to existing QA pairs JSON (skips synthesis)")
    parser.add_argument("--save", help="Path to save generated QA pairs JSON")
    parser.add_argument("--n", type=int, default=20, help="Number of QA pairs to generate (default 20)")
    args = parser.parse_args()

    model = LocalVLLM()

    if args.dataset:
        print(f"Loading QA pairs from {args.dataset}")
        with open(args.dataset) as f:
            qa_pairs = json.load(f)
    else:
        print("Fetching chunks from ChromaDB…")
        chunks = fetch_chunks(args.user_id)
        print(f"Found {len(chunks)} chunks. Synthesizing {args.n} QA pairs…")
        qa_pairs = synthesize(chunks, model, args.n)
        print(f"Generated {len(qa_pairs)} QA pairs.")

    if args.save:
        with open(args.save, "w") as f:
            json.dump(qa_pairs, f, indent=2, ensure_ascii=False)
        print(f"QA pairs saved to {args.save}")

    print(f"\nRunning RAG on {len(qa_pairs)} questions…")
    test_cases = build_test_cases(qa_pairs, args.user_id)

    metrics = [
        FaithfulnessMetric(model=model, threshold=0.7),
        AnswerRelevancyMetric(model=model, threshold=0.7),
        ContextualPrecisionMetric(model=model, threshold=0.7),
        ContextualRecallMetric(model=model, threshold=0.7),
    ]

    print("\nEvaluating…")
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
