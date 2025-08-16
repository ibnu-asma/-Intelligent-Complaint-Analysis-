import pandas as pd
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from tqdm import tqdm
import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_vector_store():
    """
    Loads cleaned data, chunks text, generates embeddings, and creates a FAISS vector store.

    The process involves:
    1. Loading the cleaned dataset from 'data/processed/filtered_complaints.csv'.
    2. Chunking the complaint narratives using RecursiveCharacterTextSplitter.
    3. Generating embeddings for the chunks using 'sentence-transformers/all-MiniLM-L6-v2'.
    4. Creating a FAISS vector store from the embeddings and documents.
    5. Saving the vector store to 'vector_store/faiss_index'.
    """
    try:
        # 1. Load cleaned dataset
        logging.info("Loading cleaned dataset...")
        df = pd.read_csv('data/processed/filtered_complaints.csv')
        # Ensure 'cleaned_narrative' column is string type and handle missing values
        df['cleaned_narrative'] = df['cleaned_narrative'].astype(str).fillna('')
        logging.info(f"Loaded {len(df)} records.")

        # 2. Implement text chunking
        logging.info("Chunking complaint narratives...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        
        documents = []
        for index, row in tqdm(df.iterrows(), total=len(df)):
            chunks = text_splitter.split_text(row['cleaned_narrative'])
            for chunk in chunks:
                documents.append(Document(
                    page_content=chunk,
                    metadata={
                        'complaint_id': row['Complaint ID'],
                        'product': row['Product']
                    }
                ))
        logging.info(f"Created {len(documents)} text chunks.")

        # 3. Generate embeddings
        logging.info("Generating embeddings using 'all-MiniLM-L6-v2'...")
        model_name = 'sentence-transformers/all-MiniLM-L6-v2'
        device = "cuda" if torch.cuda.is_available() else "cpu"
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
            encode_kwargs={"batch_size": 64}
        )
        
        # 4. Create a FAISS vector store
        logging.info("Creating FAISS vector store...")
        vector_store = FAISS.from_documents(documents, embeddings)
        logging.info("FAISS vector store created successfully.")

        # 6. Save vector store
        logging.info("Saving vector store to 'vector_store/faiss_index'...")
        vector_store.save_local('vector_store/faiss_index')
        logging.info("Vector store saved successfully.")

    except FileNotFoundError:
        logging.error("Error: The file 'data/processed/filtered_complaints.csv' was not found.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    create_vector_store()
