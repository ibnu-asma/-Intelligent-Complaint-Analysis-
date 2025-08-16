import logging
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import torch
import pandas as pd
from typing import List, Tuple, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PROMPT_TEMPLATE = (
    "You are a financial analyst assistant for CrediTrust. "
    "Answer questions about customer complaints using only the provided context. "
    "If the context lacks the answer, state so.\n"
    "Context: {context}\nQuestion: {question}\nAnswer:"
)

class RAGPipeline:
    """
    Retrieval-Augmented Generation (RAG) pipeline for answering questions about customer complaints.
    """
    def __init__(self, 
                 vector_store_path: str, 
                 embedding_model_name: str = 'sentence-transformers/all-MiniLM-L6-v2', 
                 llm_model_name: str = 'google/flan-t5-base', 
                 device: str = None) -> None:
        """
        Initialize the RAG pipeline with vector store, embedding model, and LLM.
        """
        self.vector_store_path = vector_store_path
        self.embedding_model_name = embedding_model_name
        self.llm_model_name = llm_model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs={"device": self.device},
            encode_kwargs={"batch_size": 16}
        )
        self.vector_store = FAISS.load_local(
            self.vector_store_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        self.llm = pipeline("text2text-generation", model=self.llm_model_name, device=0 if self.device == "cuda" else -1)
        logging.info(f"RAG pipeline initialized with device: {self.device}")

    def retrieve(self, question: str, k: int = 5) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Retrieve top-k relevant chunks for a given question.
        Args:
            question (str): The user question.
            k (int): Number of chunks to retrieve.
        Returns:
            Tuple[str, List[Dict[str, Any]]]: (context string, list of source metadata)
        """
        docs = self.vector_store.similarity_search(question, k=k)
        context = "\n".join([doc.page_content for doc in docs])
        sources = [doc.metadata for doc in docs]
        return context, sources

    def format_prompt(self, context: str, question: str) -> str:
        """
        Format the prompt for the LLM using the context and question.
        Args:
            context (str): Retrieved context.
            question (str): User question.
        Returns:
            str: Formatted prompt.
        """
        return PROMPT_TEMPLATE.format(context=context, question=question)

    def generate_answer(self, prompt: str) -> str:
        """
        Generate an answer using the LLM and the provided prompt.
        Args:
            prompt (str): The prompt to send to the LLM.
        Returns:
            str: Generated answer.
        """
        result = self.llm(prompt, max_new_tokens=128, truncation=True)
        return result[0]["generated_text"].strip() if isinstance(result, list) else result["generated_text"].strip()

    def answer_question(self, question: str, k: int = 5) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Retrieve context and generate an answer for a user question.
        Args:
            question (str): The user question.
            k (int): Number of chunks to retrieve.
        Returns:
            Tuple[str, List[Dict[str, Any]]]: (answer, list of source metadata)
        """
        context, sources = self.retrieve(question, k)
        prompt = self.format_prompt(context, question)
        answer = self.generate_answer(prompt)
        return answer, sources

def evaluate_rag_pipeline(rag_pipeline: RAGPipeline, questions: List[str]) -> pd.DataFrame:
    """
    Evaluate the RAG pipeline with a list of questions.
    Args:
        rag_pipeline (RAGPipeline): The RAG pipeline instance.
        questions (List[str]): List of questions to evaluate.
    Returns:
        pd.DataFrame: Results as a markdown table.
    """
    results = []
    for q in questions:
        answer, sources = rag_pipeline.answer_question(q)
        source_str = "; ".join([f"ID: {s.get('complaint_id', '')}, Product: {s.get('product', '')}" for s in sources[:2]])
        results.append({
            "Question": q,
            "Generated Answer": answer,
            "Retrieved Sources": source_str,
            "Quality Score": "",
            "Comments": ""
        })
    df = pd.DataFrame(results)
    md = df.to_markdown(index=False)
    print("\nRAG Pipeline Evaluation Results:\n")
    print(md)
    return df

def main() -> None:
    """
    Main entry point for RAG pipeline evaluation.
    """
    logging.info("Initializing RAG pipeline...")
    rag = RAGPipeline(vector_store_path="vector_store/faiss_index")
    questions = [
        "What are common issues with Buy now, pay later?",
        "How do customers describe problems with credit cards?",
        "Are there complaints about money transfers?",
        "What is a typical complaint about personal loans?",
        "Do customers mention savings account issues?",
        "What is the most frequent complaint about virtual currency?",
        "Are there complaints without narratives?",
        "How do customers feel about customer service?",
        "What are the most common complaint categories?",
        "Is there a trend in complaints over time?"
    ]
    evaluate_rag_pipeline(rag, questions)

if __name__ == "__main__":
    main()
