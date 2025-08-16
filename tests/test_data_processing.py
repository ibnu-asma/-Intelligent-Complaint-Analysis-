import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = text.replace('i am writing to file a complaint', '')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def test_clean_text_basic():
    assert clean_text("I am writing to file a complaint about XYZ!") == "about xyz"
    assert clean_text("Hello, WORLD!!") == "hello world"
    assert clean_text("   Extra   spaces   ") == "extra spaces"
    assert clean_text("i am writing to file a complaint I am writing to file a complaint") == ""
