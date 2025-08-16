"""
Unit tests for the Gradio interface in `src.app`.

This script contains unit tests for the helper functions in the `src.app` module.
The tests use the `pytest` framework and include docstrings to explain the purpose
of each test.

"""

import pytest
from src.app import chatbot_interface, clear_fn
from src.rag_pipeline import RAGPipeline

# Initialize RAGPipeline for testing purposes
# This assumes a faiss_index exists for the tests to run.
# In a real scenario, you might mock this dependency.
rag_pipeline_instance_test = RAGPipeline(vector_store_path="vector_store/faiss_index")

def test_chatbot_interface():
    """
    Test the `chatbot_interface` function.

    This test checks if the function returns a tuple with two elements (answer and sources).
    """
    question = "What are the common issues with credit cards?"
    answer, sources = chatbot_interface(question)
    assert isinstance(answer, str)
    assert isinstance(sources, str)


def test_chatbot_interface_empty_question():
    """
    Test the `chatbot_interface` function with an empty question.

    This test checks if the function handles empty questions gracefully.
    """
    question = ""
    answer, sources = chatbot_interface(question)
    assert isinstance(answer, str)
    assert isinstance(sources, str)


def test_clear_fn():
    assert clear_fn() == ("", "", "")
