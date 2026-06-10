"""
translator.py
─────────────
Translation logic using Helsinki-NLP/opus-mt-en-te (English → Telugu).
The model is cached in Streamlit's session so it loads only once.
"""

import streamlit as st
from transformers import MarianMTModel, MarianTokenizer

MODEL_NAME = "Helsinki-NLP/opus-mt-en-te"


@st.cache_resource(show_spinner="Loading translation model… (first run only)")
def _load_model():
    """Download & cache the MarianMT model and tokeniser."""
    tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
    model = MarianMTModel.from_pretrained(MODEL_NAME)
    return tokenizer, model


def translate_text(text: str) -> str:
    """
    Translate a single English string to Telugu.

    Parameters
    ----------
    text : str
        Source English text.

    Returns
    -------
    str
        Translated Telugu text, or an error message if something goes wrong.
    """
    if not text.strip():
        return ""

    try:
        tokenizer, model = _load_model()

        # Tokenise
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )

        # Generate translation
        translated_ids = model.generate(
            **inputs,
            num_beams=4,
            length_penalty=0.6,
            early_stopping=True,
        )

        # Decode
        translated = tokenizer.decode(
            translated_ids[0],
            skip_special_tokens=True,
        )
        return translated

    except Exception as exc:
        return f"⚠️ Translation error: {exc}"


def get_model_info() -> dict:
    """Return a small metadata dict for display in the UI."""
    return {
        "model": MODEL_NAME,
        "framework": "HuggingFace Transformers (MarianMT)",
        "task": "Machine Translation  en → te",
    }
