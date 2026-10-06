import os
from pathlib import Path
from typing import Dict, List, Optional

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

from openai_config import resolve_openai_base_url


def discover_chroma_backends() -> Dict[str, Dict[str, str]]:
    """Discover available ChromaDB backends in the project directory."""
    backends: Dict[str, Dict[str, str]] = {}
    current_dir = Path(".")

    # Directories that look like ChromaDB persist folders
    chroma_dirs = [
        d
        for d in current_dir.iterdir()
        if d.is_dir() and ("chroma" in d.name.lower())
    ]

    for chroma_dir in chroma_dirs:
        try:
            client = chromadb.PersistentClient(
                path=str(chroma_dir),
                settings=Settings(anonymized_telemetry=False),
            )
            collections = client.list_collections()

            for collection in collections:
                key = f"{chroma_dir.name}::{collection.name}"
                try:
                    count = collection.count()
                except Exception:
                    count = "N/A"

                backends[key] = {
                    "directory": str(chroma_dir.resolve()),
                    "collection_name": collection.name,
                    "display_name": f"{chroma_dir.name} / {collection.name} ({count} docs)",
                    "document_count": str(count),
                }

        except Exception as e:
            error_text = str(e)[:80]
            key = f"{chroma_dir.name}::error"
            backends[key] = {
                "directory": str(chroma_dir.resolve()),
                "collection_name": "",
                "display_name": f"{chroma_dir.name} (error: {error_text})",
                "document_count": "0",
            }

    return backends


def initialize_rag_system(chroma_dir: str, collection_name: str):
    """Initialize the RAG system with specified backend.

    Returns:
        (collection, success, error_message)
    """
    try:
        api_key = os.getenv("CHROMA_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        client = chromadb.PersistentClient(
            path=chroma_dir,
            settings=Settings(anonymized_telemetry=False),
        )

        if api_key:
            ef_kwargs = {
                "api_key": api_key,
                "model_name": "text-embedding-3-small",
            }
            base_url = resolve_openai_base_url(api_key)
            if base_url:
                ef_kwargs["api_base"] = base_url
            embedding_function = OpenAIEmbeddingFunction(**ef_kwargs)
            collection = client.get_collection(
                name=collection_name,
                embedding_function=embedding_function,
            )
        else:
            collection = client.get_collection(name=collection_name)

        return collection, True, None
    except Exception as e:
        return None, False, str(e)


def retrieve_documents(
    collection,
    query: str,
    n_results: int = 3,
    mission_filter: Optional[str] = None,
) -> Optional[Dict]:
    """Retrieve relevant documents from ChromaDB with optional mission filtering."""
    where_filter = None
    if mission_filter and mission_filter.lower() not in {"all", "none", ""}:
        # Normalize common UI labels to stored metadata values
        normalized = mission_filter.strip().lower().replace(" ", "_").replace("-", "_")
        aliases = {
            "apollo11": "apollo_11",
            "apollo_11": "apollo_11",
            "apollo13": "apollo_13",
            "apollo_13": "apollo_13",
            "challenger": "challenger",
        }
        mission_value = aliases.get(normalized, normalized)
        where_filter = {"mission": mission_value}

    # Cap n_results to collection size to avoid Chroma errors on small DBs
    try:
        available = collection.count()
        if available == 0:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
        n_results = min(n_results, available)
    except Exception:
        pass

    query_kwargs = {
        "query_texts": [query],
        "n_results": n_results,
        "include": ["documents", "metadatas", "distances"],
    }
    if where_filter is not None:
        query_kwargs["where"] = where_filter

    results = collection.query(**query_kwargs)

    # Deduplicate by document text while preserving similarity order (already sorted)
    if results and results.get("documents") and results["documents"][0]:
        seen = set()
        deduped_docs = []
        deduped_metas = []
        deduped_dists = []
        docs = results["documents"][0]
        metas = results.get("metadatas", [[]])[0] or [{}] * len(docs)
        dists = results.get("distances", [[]])[0] or [None] * len(docs)

        for doc, meta, dist in zip(docs, metas, dists):
            key = (doc or "").strip()
            if not key or key in seen:
                continue
            seen.add(key)
            deduped_docs.append(doc)
            deduped_metas.append(meta or {})
            deduped_dists.append(dist)

        results["documents"] = [deduped_docs]
        results["metadatas"] = [deduped_metas]
        results["distances"] = [deduped_dists]

    return results


def format_context(documents: List[str], metadatas: List[Dict]) -> str:
    """Format retrieved documents into a single context string with source attribution."""
    if not documents:
        return ""

    context_parts = ["=== RETRIEVED NASA DOCUMENT CONTEXT ===", ""]
    max_chars_per_doc = 1200

    for i, (doc, metadata) in enumerate(zip(documents, metadatas), start=1):
        metadata = metadata or {}

        mission = str(metadata.get("mission", "unknown")).replace("_", " ").title()
        source = metadata.get("source") or metadata.get("file_path") or "unknown"
        category = (
            str(metadata.get("document_category", metadata.get("category", "general")))
            .replace("_", " ")
            .title()
        )

        header = f"[Source {i}] Mission: {mission} | Document: {source} | Category: {category}"
        context_parts.append(header)
        context_parts.append("-" * len(header))

        content = (doc or "").strip()
        if len(content) > max_chars_per_doc:
            content = content[:max_chars_per_doc].rstrip() + "..."
        context_parts.append(content)
        context_parts.append("")  # blank line separator

    return "\n".join(context_parts).strip()
