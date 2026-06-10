"""
translate.py
Core translation logic using Helsinki-NLP/opus-mt-en-mul
The model supports English → many languages including Telugu.
We force Telugu output by prepending the target language tag >>tel<<
"""

import time
from transformers import MarianMTModel, MarianTokenizer

MODEL_NAME = "Helsinki-NLP/opus-mt-en-mul"


def load_model():
    """Load tokenizer and model. Called once and cached by Streamlit."""
    tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
    model = MarianMTModel.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model


def translate_text(text: str, tokenizer, model) -> dict:
    """
    Translate English text to Telugu.

    Args:
        text: English input string
        tokenizer: MarianTokenizer instance
        model: MarianMTModel instance

    Returns:
        dict with keys: telugu, input_tokens, output_tokens, time_ms
    """
    # >>tel<< tag forces Marian to produce Telugu script
    tagged_text = f">>tel<< {text.strip()}"

    inputs = tokenizer(
        [tagged_text],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    input_token_count = inputs["input_ids"].shape[1]

    start = time.perf_counter()
    outputs = model.generate(
        **inputs,
        num_beams=4,
        max_length=512,
        early_stopping=True,
        no_repeat_ngram_size=3,
    )
    elapsed_ms = round((time.perf_counter() - start) * 1000)

    output_token_count = outputs.shape[1]
    translated = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {
        "telugu": translated,
        "input_tokens": input_token_count,
        "output_tokens": output_token_count,
        "time_ms": elapsed_ms,
    }
