#!/usr/bin/env python3
"""Quick end-to-end smoke test: retrieve -> LLM -> RAGAS (no Streamlit)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import rag_client
import llm_client
import ragas_evaluator


def main() -> int:
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("CHROMA_OPENAI_API_KEY")
    if not api_key:
        print("ERROR: set OPENAI_API_KEY first")
        return 1

    os.environ["CHROMA_OPENAI_API_KEY"] = api_key
    chroma_dir = str(Path("./chroma_db_openai").resolve())
    collection_name = "nasa_space_missions_text"

    collection, ok, err = rag_client.initialize_rag_system(chroma_dir, collection_name)
    if not ok:
        print(f"ERROR: could not open Chroma collection: {err}")
        return 1

    count = collection.count()
    print(f"Collection docs: {count}")
    if count == 0:
        print("ERROR: collection is empty — run embedding_pipeline.py first")
        return 1

    questions = [
        ("apollo_11", "What was the primary goal of the Apollo 11 mission?"),
        ("apollo_13", "What major emergency occurred during the Apollo 13 mission?"),
        ("challenger", "What caused the Challenger disaster?"),
    ]

    history = []
    results = []
    for mission, question in questions:
        print(f"\n=== {mission} ===")
        print(f"Q: {question}")
        docs = rag_client.retrieve_documents(
            collection, question, n_results=3, mission_filter=mission
        )
        documents = (docs or {}).get("documents", [[]])[0]
        metadatas = (docs or {}).get("metadatas", [[]])[0]
        print(f"Retrieved chunks: {len(documents)}")
        if not documents:
            # fallback without filter if that mission is missing
            docs = rag_client.retrieve_documents(collection, question, n_results=3)
            documents = (docs or {}).get("documents", [[]])[0]
            metadatas = (docs or {}).get("metadatas", [[]])[0]
            print(f"Fallback retrieved chunks: {len(documents)}")

        context = rag_client.format_context(documents, metadatas)
        answer = llm_client.generate_response(api_key, question, context, history)
        print(f"A: {answer[:500]}{'...' if len(answer) > 500 else ''}")

        scores = ragas_evaluator.evaluate_response_quality(question, answer, documents)
        print(f"Scores: {scores}")

        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})
        results.append(
            {
                "mission": mission,
                "question": question,
                "retrieved": len(documents),
                "answer_preview": answer[:300],
                "scores": scores,
            }
        )

    out = Path("smoke_test_results.json")
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")

    # Success if every question got retrieval + a non-error answer
    ok_all = all(r["retrieved"] > 0 and r["answer_preview"] for r in results)
    return 0 if ok_all else 2


if __name__ == "__main__":
    sys.exit(main())
