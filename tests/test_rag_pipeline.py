from src.rag_pipeline import PROMPT_TEMPLATE

def test_prompt_template():
    context = "This is a test context."
    question = "What is the issue?"
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    assert "This is a test context." in prompt
    assert "What is the issue?" in prompt




