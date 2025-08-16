GEMINI.md: Automating Project Structure and Scripts for Intelligent Complaint Analysis
This file provides Gemini CLI prompts to automate the creation of the project structure and scripts for the Intelligent Complaint Analysis for Financial Services project. The prompts generate the directory structure (complaint-analysis/) and required files (e.g., README.md, data_processing.py, app.py) for all four tasks, assuming the repository is cloned at C:\Users\Cyber Defense\complaint-analysis. Each task section includes a prompt to ask whether to proceed to the next task.
Creating Project Structure
Prompt: Generate a batch script to create the project directory structure and initial files:
gemini -p "Generate a Windows batch script to create the directory structure: complaint-analysis/ with subdirectories .github/workflows, data/raw, data/processed, notebooks, src, vector_store, tests, and files .github/workflows/ci.yml, notebooks/1.0-eda.ipynb, src/__init__.py, src/data_processing.py, src/rag_pipeline.py, src/app.py, tests/test_data_processing.py, tests/test_rag_pipeline.py, Dockerfile, docker-compose.yml, requirements.txt, .gitignore, README.md, GEMINI.md." > setup_project.bat

Output Handling:

Run the script: setup_project.bat
Verify directory structure and files.

Prompt for .gitignore:
gemini -p "Generate a .gitignore file for a Python project, excluding: data/, .env, *.pyc, __pycache__/, venv/, vector_store/." > .gitignore

Prompt for requirements.txt:
gemini -p "Generate a requirements.txt file with Python packages: pandas, numpy, matplotlib, seaborn, langchain, sentence-transformers, faiss-cpu, chromadb, gradio, streamlit, pytest, flake8." > requirements.txt

Prompt to Proceed:
gemini -p "Do you want to proceed to Task 1: Exploratory Data Analysis and Data Preprocessing? (Type 'yes' to continue or 'no' to stop)."

Task 1: Exploratory Data Analysis and Data Preprocessing
Goal: Create a Jupyter notebook for EDA and preprocessing of the CFPB complaint dataset.
Prompt:
gemini -p "Generate Python code for a Jupyter notebook (notebooks/1.0-eda.ipynb) to perform EDA and preprocessing on the CFPB complaint dataset. Include: 1) Load data from data/raw/cfpb_complaints.csv using pandas, 2) Analyze complaint distribution across Credit Cards, Personal Loans, BNPL, Savings Accounts, Money Transfers, 3) Calculate and visualize narrative word counts (histograms with seaborn), 4) Count complaints with/without narratives, 5) Filter for specified products and non-empty narratives, 6) Clean narratives (lowercase, remove special characters, boilerplate text like 'I am writing to file a complaint'), 7) Save cleaned dataset to data/processed/filtered_complaints.csv. Format as Jupyter notebook code blocks with markdown explanations." > notebooks/1.0-eda.ipynb

Output Handling:

Save and open notebooks/1.0-eda.ipynb in Jupyter to validate.
Commit:git add notebooks/1.0-eda.ipynb data/processed/filtered_complaints.csv
git commit -m "Add EDA and preprocessing notebook"
git push origin main



Prompt to Proceed:
gemini -p "Task 1 completed. Do you want to proceed to Task 2: Text Chunking, Embedding, and Vector Store Indexing? (Type 'yes' to continue or 'no' to stop)."

Task 2: Text Chunking, Embedding, and Vector Store Indexing
Goal: Create a script for text chunking, embedding, and indexing complaint narratives.
Prompt:
gemini -p "Generate a Python script for src/data_processing.py to: 1) Load cleaned dataset from data/processed/filtered_complaints.csv, 2) Implement text chunking using LangChain's RecursiveCharacterTextSplitter (chunk_size=500, chunk_overlap=50), 3) Generate embeddings using sentence-transformers/all-MiniLM-L6-v2, 4) Create a FAISS vector store, 5) Store embeddings with metadata (complaint ID, product category), 6) Save vector store to vector_store/faiss_index. Include docstrings and logging." > src/data_processing.py

Output Handling:

Save and test: python src/data_processing.py
Commit:git add src/data_processing.py vector_store/faiss_index
git commit -m "Add text chunking and vector store script"
git push origin main



Prompt to Proceed:
gemini -p "Task 2 completed. Do you want to proceed to Task 3: Building the RAG Core Logic and Evaluation? (Type 'yes' to continue or 'no' to stop)."

Task 3: Building the RAG Core Logic and Evaluation
Goal: Create a RAG pipeline script and evaluation table.
Prompt for rag_pipeline.py:
gemini -p "Generate a Python script for src/rag_pipeline.py to: 1) Load FAISS vector store from vector_store/faiss_index, 2) Embed user questions using sentence-transformers/all-MiniLM-L6-v2, 3) Retrieve top-5 relevant chunks via similarity search, 4) Use a prompt template: 'You are a financial analyst assistant for CrediTrust. Answer questions about customer complaints using only the provided context. If the context lacks the answer, state so. Context: {context} Question: {question} Answer:', 5) Generate responses using an LLM (e.g., Hugging Face pipeline or LangChain with Mistral), 6) Evaluate with 5-10 questions (e.g., 'What are common BNPL issues?'), creating a markdown table (Question, Generated Answer, Retrieved Sources [1-2], Quality Score [1-5], Comments). Include docstrings." > src/rag_pipeline.py

Prompt for Unit Tests:
gemini -p "Generate a Python script for tests/test_rag_pipeline.py with at least two unit tests for a helper function in src/rag_pipeline.py (e.g., question embedding or retrieval function). Use pytest and include docstrings." > tests/test_rag_pipeline.py

Output Handling:

Save and test:python src/rag_pipeline.py
pytest tests/test_rag_pipeline.py


Commit:git add src/rag_pipeline.py tests/test_rag_pipeline.py
git commit -m "Add RAG pipeline and unit tests"
git push origin main



Prompt to Proceed:
gemini -p "Task 3 completed. Do you want to proceed to Task 4: Creating an Interactive Chat Interface? (Type 'yes' to continue or 'no' to stop)."

Task 4: Creating an Interactive Chat Interface
Goal: Create a Gradio or Streamlit interface for the RAG chatbot.
Prompt for app.py:
gemini -p "Generate a Python script for src/app.py to create a Gradio interface with: 1) Text input for user questions, 2) Submit button, 3) Display area for AI-generated answers, 4) Display of source text chunks below the answer, 5) Clear button to reset conversation, 6) (Optional) Response streaming. Load the RAG pipeline from src/rag_pipeline.py. Include docstrings and ensure intuitive UI." > src/app.py

Prompt for Unit Tests:
gemini -p "Generate a Python script for tests/test_app.py with at least two unit tests for a helper function in src/app.py (e.g., answer generation or source display). Use pytest and include docstrings." > tests/test_app.py

Prompt for Dockerfile:
gemini -p "Generate a Dockerfile to set up a Python environment, install requirements.txt, and run the Gradio application with python src/app.py." > Dockerfile

Prompt for docker-compose.yml:
gemini -p "Generate a docker-compose.yml file to build and run the Gradio service from Dockerfile, exposing port 7860." > docker-compose.yml

Prompt for ci.yml:
gemini -p "Generate a GitHub Actions workflow file for .github/workflows/ci.yml to: 1) Run flake8 linting on src/ and tests/, 2) Run pytest on tests/. Fail the build if either step fails." > .github/workflows/ci.yml

Output Handling:

Save and test: docker-compose up --build
Commit:git add src/app.py tests/test_app.py Dockerfile docker-compose.yml .github/workflows/ci.yml
git commit -m "Add Gradio interface, Docker, and CI/CD files"
git push origin main



Prompt to Proceed:
gemini -p "Task 4 completed. Project deliverables complete. Would you like to review or proceed with submission? (Type 'yes' to review or 'no' to submit)."

Notes

Validation: Test each script (e.g., python src/data_processing.py) and validate outputs in notebooks/1.0-eda.ipynb.
Dataset Path: Assumes CFPB dataset at data/raw/cfpb_complaints.csv; adjust prompts if different.
Submission: Include all files in the GitHub repository for interim (06 July 2025) and final (08 July 2025) submissions.
Interactive Flow: Run the “Prompt to Proceed” commands after each task and respond with “yes” or “no” to continue or pause.
References: Use provided links (e.g., CFPB dataset, LangChain, FAISS) for guidance.
