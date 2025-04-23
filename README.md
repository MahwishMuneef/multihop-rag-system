# Multi-hop RAG System for Maternal Health

This repository contains a Retrieval-Augmented Generation (RAG) pipeline for answering complex, multi-hop questions from a collection of maternal health-related documents. The system is designed to:

- **Ingest multiple PDFs**
- **Decompose complex queries into sub-questions**
- **Retrieve relevant chunks from a Chroma vector database**
- **Use OpenAI's LLM to generate accurate, context-aware answers**
- **Evaluate performance using ROUGE metrics**

---

## 📂 Project Structure

```bash
.
├── multihop.py          # Main multi-hop question answering script
├── evaluation.py        # Script to evaluate generated answers using ROUGE
├── uploaded_docs/       # Folder containing uploaded maternal health PDFs
├── data/                # Additional document resources
├── .gitattributes       # Git LFS config (for PDFs)
├── .gitignore
├── README.md


---

## ✅ Features

- Multi-hop question decomposition using LangChain
- Document retrieval with Chroma vector DB
- OpenAI GPT for final answer generation
- ROUGE score evaluation of generated answers
- Git LFS support for large `.pdf` documents

---

## 🔧 Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/MahwishMuneef/multihop-rag-system.git
cd multihop-rag-system

### 2. Create a virtual environment and install dependencies
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

### 3. Set your environment variables
OPENAI_API_KEY=your-openai-api-key
AZURE_CONTENT_SAFETY_ENDPOINT==your-end-point
AZURE_CONTENT_SAFETY_KEY=your-api-key

## 🔧 Usage

python multihop.py


## 🔧 Run Evaluation
python evaluation.py

## 🔧 Git LFS: Large File Support
git lfs install


## 🧠 Technologies Used

Python
LangChain
ChromaDB
OpenAI API
Git LFS
ROUGE (Evaluate)


## 🤝 Contributions

Open to feedback, issues, and pull requests!

