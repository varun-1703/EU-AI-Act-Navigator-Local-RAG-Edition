# 🇪🇺 EU AI Act Navigator: Local RAG Edition

**Ask anything about the EU AI Act and get answers powered by a local, privacy-preserving Retrieval-Augmented Generation (RAG) pipeline!**

---

## 🚀 Project Overview

The **EU AI Act Navigator: Local RAG Edition** is an interactive, local-first application that leverages Retrieval-Augmented Generation (RAG) to answer questions about the European Union's AI Act. It combines state-of-the-art open-source language models (TinyLlama) with semantic search over the official Act text, all running on your own hardware—no cloud required!

---

## 🛠️ Technical Stack

- **Language Model:** [TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) (runs locally, 4-bit quantized for efficiency)
- **Embeddings:** `sentence-transformers/all-mpnet-base-v2` (for semantic chunk retrieval)
- **Vector Store:** FAISS (fast, in-memory similarity search)
- **Frameworks:** 
  - [Streamlit](https://streamlit.io/) (modern, interactive UI)
  - [LangChain](https://python.langchain.com/) (RAG pipeline, prompt management)
  - [Transformers](https://huggingface.co/docs/transformers/index) (model loading, tokenization)
- **Hardware:** Optimized for CUDA GPUs, but falls back to CPU if needed

---

## 📂 Project Structure

```
.
├── app.py                      # Streamlit app: RAG pipeline, chat UI
├── 01-Data_Processing.ipynb    # Notebook: PDF loading, chunking, embedding, FAISS index creation
├── 02-Local_LLM_Setup.ipynb    # Notebook: LLM (TinyLlama) setup, quantization, testing
├── 03-RAG_Chain.ipynb          # Notebook: End-to-end RAG chain, integration, and testing
├── data/
│   └── EU_AI_Act_latest.pdf    # (Not tracked) Official EU AI Act PDF
├── vectorstore/
│   └── db_faiss_eu_ai_act/     # (Not tracked) FAISS index and metadata
├── venv/                       # (Not tracked) Python virtual environment
└── .gitignore                  # Excludes data, vectorstore, venv, and temp files
```

---

## 🧑‍💻 How It Works

1. **Data Processing** (`01-Data_Processing.ipynb`)
   - Loads the official EU AI Act PDF.
   - Splits the document into ~1000-character overlapping chunks for context-rich retrieval.
   - Embeds each chunk using a high-quality sentence transformer.
   - Stores embeddings in a FAISS vector database for fast similarity search.

2. **LLM Setup** (`02-Local_LLM_Setup.ipynb`)
   - Downloads and configures TinyLlama for local inference.
   - Applies 4-bit quantization (via BitsAndBytes) for efficient GPU/CPU usage.
   - Verifies model and tokenizer loading, and runs test generations.

3. **RAG Chain** (`03-RAG_Chain.ipynb` & `app.py`)
   - On user question, retrieves the most relevant chunks from the Act using FAISS.
   - Constructs a prompt with retrieved context and the user's question.
   - TinyLlama generates a concise, context-grounded answer.
   - Streamlit provides a chat-like interface for seamless interaction.

---

## ⚡ Features

- **Local-Only:** No data leaves your machine. Full privacy.
- **GPU-Accelerated:** Runs efficiently on consumer GPUs (4GB+ VRAM), but supports CPU fallback.
- **Semantic Search:** Finds the most relevant legal text for your query.
- **RAG Pipeline:** Combines retrieval and generation for accurate, context-aware answers.
- **Modern UI:** Chat interface via Streamlit.

---

## 🏗️ Setup & Usage

1. **Clone the repo and set up a virtual environment:**
   ```bash
   git clone <your-repo-url>
   cd eu_ai_act_navigator
   python -m venv venv
   venv\\Scripts\\activate  # On Windows
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the official EU AI Act PDF:**
   - Place it as `data/EU_AI_Act_latest.pdf`.

4. **Run the data processing notebook:**
   - Open `01-Data_Processing.ipynb` in Jupyter/VSCode.
   - Execute all cells to generate the FAISS vectorstore.

5. **Run the app:**
   ```bash
   streamlit run app.py
   ```

---

## 📝 Technical Notes

- **Model Quantization:** Uses 4-bit quantization for TinyLlama via BitsAndBytes, enabling fast inference on limited VRAM.
- **Embeddings:** All-mpnet-base-v2 is used for high-quality semantic chunking and retrieval.
- **FAISS:** Vectorstore is not tracked in git due to size; regenerate as needed.
- **.gitignore:** Excludes data, vectorstore, venv, and temp files for a clean repo.

---

## 📢 Acknowledgements

- [TinyLlama](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
- [Sentence Transformers](https://www.sbert.net/)
- [LangChain](https://python.langchain.com/)
- [Streamlit](https://streamlit.io/)
- [FAISS](https://github.com/facebookresearch/faiss)

---

## 💡 Future Ideas

- Add support for more legal documents.
- Enable multi-lingual Q&A.
- Dockerize for easy deployment.

---

**Empowering transparent, local AI for legal compliance!** 