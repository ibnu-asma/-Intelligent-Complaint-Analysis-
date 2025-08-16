import gradio as gr
import logging
from rag_pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the RAG pipeline (on app startup)
logging.info("Loading RAG pipeline for Gradio app...")
rag = RAGPipeline(vector_store_path="vector_store/faiss_index")

def chat(query):
    """
    Handle a user query: retrieve answer and sources from the RAG pipeline.
    Returns the answer and the top-2 source chunks.
    """
    if not query.strip():
        return "Please enter a question.", ""
    answer, sources = rag.answer_question(query, k=2)
    sources_str = "\n\n".join([
        f"Source {i+1}: [ID: {s.get('complaint_id', '')}, Product: {s.get('product', '')}]" for i, s in enumerate(sources[:2])
    ])
    return answer, sources_str

def clear_fn():
    """Clear the chat input and outputs."""
    return "", "", ""

def main():
    """
    Launch the Gradio app for the RAG chatbot.
    """
    with gr.Blocks() as demo:
        gr.Markdown("""
        # CrediTrust Complaint Analysis Chatbot
        Ask any question about customer complaints. The AI will answer using only the complaint database.
        """)
        with gr.Row():
            question = gr.Textbox(label="Your Question", placeholder="Type your question here...")
        with gr.Row():
            answer = gr.Textbox(label="AI Answer", interactive=False)
        with gr.Row():
            sources = gr.Textbox(label="Source Chunks", interactive=False)
        with gr.Row():
            submit_btn = gr.Button("Submit")
            clear_btn = gr.Button("Clear")
        submit_btn.click(chat, inputs=question, outputs=[answer, sources])
        clear_btn.click(clear_fn, inputs=None, outputs=[question, answer, sources])
    demo.launch()

if __name__ == "__main__":
    main()
