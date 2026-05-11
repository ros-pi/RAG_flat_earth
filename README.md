# 🌍 Flat Earth RAG Engine (Pure Python)

A lightweight, highly resilient Retrieval-Augmented Generation (RAG) pipeline built entirely in Python. This project demonstrates how to securely ground a Large Language Model in custom, private data to eliminate hallucinations, even in a topic highly contradictive to global knowledge.

To prove the success of the context injection, this engine uses "Flat Earth" lore. Because LLMs are trained to know the Earth is round, successfully forcing the LLM to answer using Flat Earth theory proves that the RAG pipeline has successfully overridden the model's default training data.

## 🚀 Features
* **Zero C++ Dependencies:** Uses a pure-Python `InMemoryVectorStore`, completely bypassing the common Windows SQLite/C++ segmentation faults associated with ChromaDB.
* **Modern LCEL Architecture:** Built using LangChain Expression Language for a clean, declarative data pipeline.
* **Real-time Streaming:** Streams the LLM's response word-by-word to the terminal, mimicking a production-grade chat UI.
* **Anti-Hallucination Guardrails:** Strict prompt engineering ensures the model refuses to answer queries outside the provided context.

## 🏗️ Architecture
1. **Ingestion:** `TextLoader` reads the custom dataset (data/flat_earth_lore.txt).
2. **Chunking:** `RecursiveCharacterTextSplitter` breaks text into 300-character overlapping chunks.
3. **Embedding:** `gemini-embedding-001` converts text into dense vectors.
4. **Retrieval:** Semantic search retrieves the top 3 most relevant chunks based on the user's query.
5. **Generation:** `gemini-2.5-flash` receives the strict prompt and generates the streamed answer.
