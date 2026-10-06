# NASA Mission Intelligence RAG — Project Report

## Overview
This project implements a Retrieval-Augmented Generation (RAG) system for NASA mission documents (Apollo 11, Apollo 13, Challenger). Users ask questions in a Streamlit chat UI; the system retrieves relevant document chunks from ChromaDB, generates grounded answers with an OpenAI LLM, and scores each answer with RAGAS metrics in real time.

## Components Implemented
1. **`llm_client.py`** — OpenAI chat client with a NASA-expert system prompt, retrieved-context injection, conversation history, and instructions to avoid ungrounded claims.
2. **`rag_client.py`** — ChromaDB discovery, semantic retrieval with configurable top-k and optional mission metadata filter, context formatting with source attribution and deduplication.
3. **`embedding_pipeline.py`** — Text chunking (`chunk_size` / `chunk_overlap`), OpenAI embeddings, metadata (`source`, `mission`, category, etc.), Chroma persistence, CLI flags including `--update-mode` (`skip` | `update` | `replace`) and `--stats-only`.
4. **`ragas_evaluator.py`** — Response Relevancy + Faithfulness (+ BLEU/ROUGE), input validation, and batch evaluation over `evaluation_dataset.txt`.
5. **`chat.py`** — Streamlit interface wiring RAG + LLM + evaluation, with model selection and mission filter.

## Challenges and Solutions
| Challenge | Solution |
|---|---|
| Windows path issues cloning Udacity monorepo (trailing spaces in some lesson folders) | Sparse-checkout of Windows-safe folders; project starter is fully available |
| Matching embeddings at index time and query time | Use `text-embedding-3-small` via OpenAI for both pipeline writes and Chroma query embedding function |
| Update modes for re-processing files | Implement `skip` / `update` / `replace` against stable IDs `mission_source_chunk_####` |
| RAGAS import / VertexAI compatibility | Follow README pin versions; install `langchain-google-vertexai` as listed in `requirements.txt` |
| Empty or bad evaluation inputs | Validate question/answer/contexts and return a clear `error` dict instead of crashing |

## How to Run
```bash
cd Project-NASA-Mission-Intelligence-Starter
pip install -r requirements.txt
set OPENAI_API_KEY=your-key   # PowerShell: $env:OPENAI_API_KEY="your-key"
# Vocareum course keys (voc-...) automatically use https://openai.vocareum.com/v1

# 1) Build embeddings
python embedding_pipeline.py --openai-key %OPENAI_API_KEY% --data-path ./data_text --chroma-dir ./chroma_db_openai

# 2) Inspect collection
python embedding_pipeline.py --openai-key %OPENAI_API_KEY% --chroma-dir ./chroma_db_openai --stats-only

# 3) Chat UI
streamlit run chat.py

# 4) Optional batch evaluation smoke test
python ragas_evaluator.py --dataset evaluation_dataset.txt
```

## Evaluation Dataset
`evaluation_dataset.txt` includes 8 questions across overview, crew, timeline, emergency, technical, and disaster-analysis categories for Apollo 11, Apollo 13, and Challenger.

## Sample Questions Used in Testing
1. What was the primary goal of the Apollo 11 mission?
2. What major emergency occurred during the Apollo 13 mission?
3. What caused the Space Shuttle Challenger disaster, and when did it occur?
4. Who were the crew members of Apollo 11?
5. What were the names of the Apollo 11 command module and lunar module?

Expected behavior: answers cite retrieved NASA text, stay within mission scope when a filter is selected, and display Relevancy / Faithfulness scores in the sidebar when evaluation is enabled.
