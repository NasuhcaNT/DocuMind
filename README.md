# SQuAD-RAG Engine 🚀

This project is a high-performance **Retrieval-Augmented Generation (RAG)** prototype. It connects a Large Language Model to the **SQuAD (Stanford Question Answering Dataset)** to provide fact-based answers using semantic search.

## 🛠️ Architecture (The 4 Pillars)
1. **Retrieval:** Uses `ChromaDB` and `all-MiniLM-L6-v2` embeddings to find relevant context from the SQuAD dataset.
2. **Augmentation:** Dynamically constructs prompts by injecting retrieved context.
3. **Generation:** Leverages `HuggingFace` models to generate human-like, accurate responses.
4. **Evaluation:** Includes a custom `truncate_documents` function to optimize context relevance and manual evaluation loops.

## 📋 How It Works
The system follows the **LCEL (LangChain Expression Language)** logic:
`Question -> Vector Search -> Context Injection -> LLM Inference -> Clean Output`

## ⚙️ Setup
1. Clone the repo:
   ```bash
   git clone [https://github.com/NasuhcaNT/DocuMind.git](https://github.com/NasuhcaNT/DocuMind.git)