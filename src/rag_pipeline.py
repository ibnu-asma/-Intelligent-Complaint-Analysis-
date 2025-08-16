import logging
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import torch
import pandas as pd
import os

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
    def __init__(self, vector_store_path, embedding_model_name='sentence-transformers/all-MiniLM-L6-v2', llm_model_name='google/flan-t5-base', device=None):
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

    def retrieve(self, question, k=5):
        """
        Retrieve top-k relevant chunks for a given question.
        Returns: (context, sources)
        """
        docs = self.vector_store.similarity_search(question, k=k)
        context = "\n".join([doc.page_content for doc in docs])
        sources = [doc.metadata for doc in docs]
        return context, sources

    def generate_answer(self, question, context):
        """
        Generate an answer using the LLM and the provided context.
        """
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)
        result = self.llm(prompt, max_new_tokens=128, truncation=True)
        return result[0]["generated_text"].strip() if isinstance(result, list) else result["generated_text"].strip()

    def answer_question(self, question, k=5):
        context, sources = self.retrieve(question, k)
        answer = self.generate_answer(question, context)
        return answer, sources

def evaluate_rag_pipeline(rag_pipeline, questions):
    """
    Evaluate the RAG pipeline with a list of questions.
    Returns a markdown table with columns: Question, Generated Answer, Retrieved Sources, Quality Score, Comments.
    """
    results = []
    for q in questions:
        answer, sources = rag_pipeline.answer_question(q)
        # For now, leave Quality Score and Comments blank for manual review
        source_str = "; ".join([f"ID: {s.get('complaint_id', '')}, Product: {s.get('product', '')}" for s in sources[:2]])
        results.append({
            "Question": q,
            "Generated Answer": answer,
            "Retrieved Sources": source_str,
            "Quality Score": "",
            "Comments": ""
        })
    # Output as markdown table
    df = pd.DataFrame(results)
    md = df.to_markdown(index=False)
    print("\nRAG Pipeline Evaluation Results:\n")
    print(md)
    return df

def main():
    logging.info("Initializing RAG pipeline...")
    rag = RAGPipeline(vector_store_path="vector_store/faiss_index")
    # Example evaluation questions
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
