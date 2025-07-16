# 🚀 Code Generation using RAG (Retrieval-Augmented Generation)

> A modular AI system that takes a natural language programming task and generates Python code by retrieving similar examples from the HumanEval dataset and using a code generation LLM.

---

## 📌 Objective

This project demonstrates how to combine **semantic retrieval** and **code generation** into a Retrieval-Augmented Generation (RAG) pipeline.  
Given a programming task described in plain English, the system:

1. Retrieves the **most similar tasks** and solutions from the [HumanEval dataset](https://github.com/openai/human-eval).
2. Uses an **open-source LLM** to generate a Python function based on the task and the retrieved examples.

---

## 🛠️ Features

- ✅ Load and process the HumanEval dataset
- 🔍 Generate semantic embeddings using `sentence-transformers`
- 💾 Store vectors in a vector database (`ChromaDB`)
- 🔎 Retrieve top-K similar prompts using vector similarity search
- ✍️ Generate Python code using an open-source LLM
- 🔁 Clean modular RAG pipeline design

---


## 🧠 Models Used

| Task               | Model                                      |
|--------------------|--------------------------------------------|
| Text Embedding     | `all-MiniLM-L6-v2` (via sentence-transformers) |
| Code Generation    | Code-aware LLM (e.g., CodeGen, StarCoder, CodeLlama) |

---

## 📦 Installation

```bash
pip install sentence-transformers chromadb pandas openai transformers


