import pandas as pd
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from tqdm import tqdm
import torch
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataProcessor:
    """
    Handles loading, chunking, embedding, and vector store creation for complaint narratives.
    """
    def __init__(self, 
                 input_path: str = 'data/processed/filtered_complaints.csv',
                 vector_store_path: str = 'vector_store/faiss_index',
                 chunk_size: int = 500,
                 chunk_overlap: int = 50,
                 embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2',
                 batch_size: int = 64) -> None:
        """
        Initialize the DataProcessor.
        """
        self.input_path = input_path
        self.vector_store_path = vector_store_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_data(self) -> pd.DataFrame:
        """
        Load the cleaned complaints dataset.
        Returns:
            pd.DataFrame: Loaded DataFrame.
        """
        logging.info(f"Loading cleaned dataset from {self.input_path}...")
        df = pd.read_csv(self.input_path)
        df['cleaned_narrative'] = df['cleaned_narrative'].astype(str).fillna('')
        logging.info(f"Loaded {len(df)} records.")
        return df

    def chunk_narratives(self, df: pd.DataFrame) -> List[Document]:
        """
        Chunk complaint narratives using RecursiveCharacterTextSplitter.
        Args:
            df (pd.DataFrame): DataFrame with 'cleaned_narrative'.
        Returns:
            List[Document]: List of chunked Documents with metadata.
        """
        logging.info("Chunking complaint narratives...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        documents = []
        for _, row in tqdm(df.iterrows(), total=len(df)):
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
        return documents

    def create_embeddings(self) -> HuggingFaceEmbeddings:
        """
        Create a HuggingFaceEmbeddings object for the specified model and device.
        Returns:
            HuggingFaceEmbeddings: Embedding model instance.
        """
        logging.info(f"Generating embeddings using '{self.embedding_model}' on {self.device}...")
        return HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            model_kwargs={"device": self.device},
            encode_kwargs={"batch_size": self.batch_size}
        )

    def create_vector_store(self, documents: List[Document], embeddings: HuggingFaceEmbeddings) -> FAISS:
        """
        Create a FAISS vector store from documents and embeddings.
        Args:
            documents (List[Document]): List of chunked documents.
            embeddings (HuggingFaceEmbeddings): Embedding model.
        Returns:
            FAISS: The created FAISS vector store.
        """
        logging.info("Creating FAISS vector store...")
        vector_store = FAISS.from_documents(documents, embeddings)
        logging.info("FAISS vector store created successfully.")
        return vector_store

    def save_vector_store(self, vector_store: FAISS) -> None:
        """
        Save the FAISS vector store to disk.
        Args:
            vector_store (FAISS): The FAISS vector store to save.
        """
        logging.info(f"Saving vector store to '{self.vector_store_path}'...")
        vector_store.save_local(self.vector_store_path)
        logging.info("Vector store saved successfully.")

    def process(self) -> None:
        """
        Complete pipeline: load data, chunk, embed, create and save vector store.
        """
        try:
            df = self.load_data()
            documents = self.chunk_narratives(df)
            embeddings = self.create_embeddings()
            vector_store = self.create_vector_store(documents, embeddings)
            self.save_vector_store(vector_store)
        except FileNotFoundError:
            logging.error(f"Error: The file '{self.input_path}' was not found.")
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    processor = DataProcessor()
    processor.process()
