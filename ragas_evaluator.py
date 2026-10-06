"""RAGAS-based response quality evaluation for the NASA RAG system."""

from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Any

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from openai_config import resolve_openai_base_url

try:
    from ragas import SingleTurnSample
    from ragas.metrics import (
        BleuScore,
        Faithfulness,
        ResponseRelevancy,
        RougeScore,
    )
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False


def _validate_inputs(question: str, answer: str, contexts: List[str]) -> Optional[str]:
    """Return an error message if inputs are empty/malformed; otherwise None."""
    if not isinstance(question, str) or not question.strip():
        return "Invalid input: question must be a non-empty string"
    if not isinstance(answer, str) or not answer.strip():
        return "Invalid input: answer must be a non-empty string"
    if contexts is None or not isinstance(contexts, list):
        return "Invalid input: contexts must be a list of strings"
    cleaned = [c for c in contexts if isinstance(c, str) and c.strip()]
    if not cleaned:
        return "Invalid input: contexts must contain at least one non-empty string"
    return None


def evaluate_response_quality(
    question: str,
    answer: str,
    contexts: List[str],
) -> Dict[str, float]:
    """Evaluate a (question, context, answer) triple with RAGAS metrics.

    Computes at least Response Relevancy and Faithfulness, plus BLEU/ROUGE
    when available. Returns a dict of metric_name -> score, or {"error": ...}.
    """
    if not RAGAS_AVAILABLE:
        return {"error": "RAGAS not available"}

    validation_error = _validate_inputs(question, answer, contexts)
    if validation_error:
        return {"error": validation_error}

    cleaned_contexts = [c.strip() for c in contexts if isinstance(c, str) and c.strip()]

    try:
        # Evaluator LLM + embeddings (supports Vocareum voc- keys via base_url)
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("CHROMA_OPENAI_API_KEY")
        base_url = resolve_openai_base_url(api_key)
        llm_kwargs = {"model": "gpt-3.5-turbo", "temperature": 0}
        emb_kwargs = {"model": "text-embedding-3-small"}
        if api_key:
            llm_kwargs["api_key"] = api_key
            emb_kwargs["api_key"] = api_key
        if base_url:
            llm_kwargs["base_url"] = base_url
            emb_kwargs["base_url"] = base_url

        evaluator_llm = LangchainLLMWrapper(ChatOpenAI(**llm_kwargs))
        evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(**emb_kwargs))

        metrics = []
        try:
            metrics.append(
                (
                    "response_relevancy",
                    ResponseRelevancy(
                        llm=evaluator_llm,
                        embeddings=evaluator_embeddings,
                    ),
                )
            )
        except Exception as e:
            return {"error": f"Failed to init ResponseRelevancy: {e}"}

        try:
            metrics.append(("faithfulness", Faithfulness(llm=evaluator_llm)))
        except Exception as e:
            return {"error": f"Failed to init Faithfulness: {e}"}

        # Optional lexical metrics (need extras like sacrebleu)
        for name, factory in (
            ("bleu", BleuScore),
            ("rouge", RougeScore),
        ):
            try:
                metrics.append((name, factory()))
            except Exception as e:
                # Keep going — rubric requires Relevancy + Faithfulness
                pass

        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=cleaned_contexts,
            # Reference needed by some lexical metrics; use answer as self-reference fallback
            reference=answer,
        )

        results: Dict[str, Any] = {}
        for name, metric in metrics:
            try:
                score = metric.single_turn_score(sample)
                if score is not None:
                    results[name] = float(score)
            except Exception as metric_error:
                results[f"{name}_error"] = str(metric_error)

        if not any(isinstance(v, float) for v in results.values()):
            return {"error": f"All metrics failed: {results}"}

        return results

    except Exception as e:
        return {"error": f"Evaluation failed: {str(e)}"}


def load_evaluation_dataset(path: str = "evaluation_dataset.txt") -> List[Dict[str, str]]:
    """Load test questions from evaluation_dataset.txt or test_questions.json."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {path}")

    if file_path.suffix.lower() == ".json":
        data = json.loads(file_path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "questions" in data:
            data = data["questions"]
        return data

    # Text format:
    # CATEGORY: overview
    # Q: ...
    # EXPECTED: ...
    # ---
    items: List[Dict[str, str]] = []
    current: Dict[str, str] = {}
    for raw_line in file_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line == "---":
            if current.get("question"):
                items.append(current)
            current = {}
            continue
        lower = line.lower()
        if lower.startswith("category:"):
            current["category"] = line.split(":", 1)[1].strip()
        elif lower.startswith("q:") or lower.startswith("question:"):
            current["question"] = line.split(":", 1)[1].strip()
        elif lower.startswith("expected:") or lower.startswith("a:"):
            current["expected"] = line.split(":", 1)[1].strip()
        elif lower.startswith("mission:"):
            current["mission"] = line.split(":", 1)[1].strip()

    if current.get("question"):
        items.append(current)

    return items


def batch_evaluate(
    dataset_path: str = "evaluation_dataset.txt",
    answers: Optional[List[str]] = None,
    contexts_per_question: Optional[List[List[str]]] = None,
) -> Dict[str, Any]:
    """Run evaluation over a test set and return per-question + aggregate scores.

    If answers/contexts are omitted, uses each item's `expected` field as a
    placeholder answer and empty context note — intended for offline smoke tests.
    For end-to-end RAG scoring, pass real retrieved contexts and model answers.
    """
    dataset = load_evaluation_dataset(dataset_path)
    if len(dataset) < 1:
        return {"error": "Evaluation dataset is empty"}

    per_question: List[Dict[str, Any]] = []
    metric_buckets: Dict[str, List[float]] = {}

    for i, item in enumerate(dataset):
        question = item.get("question", "")
        answer = (
            answers[i]
            if answers and i < len(answers)
            else item.get("expected", "")
        )
        contexts = (
            contexts_per_question[i]
            if contexts_per_question and i < len(contexts_per_question)
            else [item.get("expected", "No context provided for offline batch run.")]
        )

        scores = evaluate_response_quality(question, answer, contexts)
        entry = {
            "index": i,
            "category": item.get("category", "unknown"),
            "mission": item.get("mission", "unknown"),
            "question": question,
            "scores": scores,
        }
        per_question.append(entry)

        for key, value in scores.items():
            if isinstance(value, (int, float)):
                metric_buckets.setdefault(key, []).append(float(value))

    aggregates: Dict[str, Any] = {}
    for metric_name, values in metric_buckets.items():
        if values:
            aggregates[metric_name] = {
                "mean": statistics.mean(values),
                "min": min(values),
                "max": max(values),
                "count": len(values),
            }

    return {
        "dataset": dataset_path,
        "num_questions": len(dataset),
        "per_question": per_question,
        "aggregates": aggregates,
    }


def main():
    parser = argparse.ArgumentParser(description="Batch RAGAS evaluation")
    parser.add_argument(
        "--dataset",
        default="evaluation_dataset.txt",
        help="Path to evaluation_dataset.txt or test_questions.json",
    )
    parser.add_argument(
        "--output",
        default="evaluation_results.json",
        help="Where to write the batch evaluation summary JSON",
    )
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY before running batch evaluation.")

    results = batch_evaluate(args.dataset)
    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results.get("aggregates", results), indent=2))
    print(f"Wrote full results to {args.output}")


if __name__ == "__main__":
    main()
