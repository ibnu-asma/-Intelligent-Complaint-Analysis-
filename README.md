# CrediTrust Intelligent Complaint Analysis

## Overview
This project provides an end-to-end pipeline for intelligent analysis and retrieval-augmented generation (RAG) over the CFPB financial complaints dataset. It enables:
- Exploratory data analysis (EDA) and preprocessing
- Text chunking, embedding, and vector store indexing
- Retrieval-augmented question answering (RAG pipeline)
- An interactive Gradio chatbot interface

## Project Structure
```
complaint-analysis/
├── .github/workflows/ci.yml         # CI/CD workflow for linting and tests
├── data/
│   ├── raw/                         # Raw data files (e.g., cfpb_complaints.csv)
│   └── processed/                   # Processed/cleaned data
├── docker-compose.yml               # Docker Compose for Gradio app
├── Dockerfile                       # Dockerfile for app containerization
├── GEMINI.md                        # Project automation and task prompts
├── notebooks/
│   └── 1.0-eda.ipynb                # EDA and preprocessing notebook
├── README.md                        # Project documentation (this file)
├── requirements.txt                 # Python dependencies
├── src/
│   ├── __init__.py                  # Package marker
│   ├── app.py                       # Gradio chatbot app
│   ├── data_processing.py           # Chunking, embedding, vector store creation
│   └── rag_pipeline.py              # RAG pipeline and evaluation
├── tests/
│   ├── test_data_processing.py      # Unit tests for data processing
│   └── test_rag_pipeline.py         # Unit tests for RAG pipeline
├── vector_store/                    # FAISS vector store files
└── venv/                            # Python virtual environment (not tracked)
```

## Setup Instructions
1. **Clone the repository**
2. **Create and activate a virtual environment**
   ```sh
   python -m venv venv
   venv\Scripts\activate  # On Windows
   # or
   source venv/bin/activate  # On macOS/Linux
   ```
3. **Install dependencies**
   ```sh
   pip install -r requirements.txt
   ```
4. **Download the CFPB complaints dataset**
   - Place the raw CSV in `data/raw/cfpb_complaints.csv`

## How to Run Each Task

### 1. Exploratory Data Analysis & Preprocessing
- Open `notebooks/1.0-eda.ipynb` in JupyterLab/VSCode
- Run all cells to generate and save the cleaned dataset to `data/processed/filtered_complaints.csv`

### 2. Text Chunking, Embedding, and Vector Store Indexing
- Run:
  ```sh
  python src/data_processing.py
  ```
- This will create a FAISS vector store in `vector_store/faiss_index`

### 3. RAG Pipeline & Evaluation
- Run:
  ```sh
  python src/rag_pipeline.py
  ```
- This will answer sample questions and print a markdown evaluation table

### 4. Interactive Chatbot (Gradio)
- Run:
  ```sh
  python src/app.py
  ```
- Open the provided local URL in your browser to chat with the AI

## Example Questions to Ask the Chatbot
- What are common issues with Buy now, pay later?
- How do customers describe problems with credit cards?
- Are there complaints about money transfers?
- What is a typical complaint about personal loans?
- Do customers mention savings account issues?
- What is the most frequent complaint about virtual currency?

## Hardware & Performance Notes
- For best performance, use a CUDA-capable GPU (the pipeline will auto-detect and use it)
- Embedding and LLM inference are much faster on GPU
- If you see warnings about deprecated imports, update to the latest `langchain-huggingface` package as needed

## Troubleshooting
- **Missing dependencies:** Run `pip install -r requirements.txt` and `pip install tabulate tqdm`
- **FAISS deserialization error:** The pipeline now sets `allow_dangerous_deserialization=True` for trusted local files
- **Model download issues:** Ensure you have a stable internet connection for Hugging Face model downloads
- **Windows symlink warning:** You can ignore this, or enable Developer Mode for optimal caching

## Credits & References
- [CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/)
- [LangChain](https://python.langchain.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Gradio](https://gradio.app/)
