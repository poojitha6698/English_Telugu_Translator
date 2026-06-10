import streamlit as st

from src.predict import Translator

st.set_page_config(
    page_title="English To Telugu Translator",
    page_icon="🌍",
    layout="centered"
)

translator = Translator()

st.markdown(
"""
<h1 style='text-align:center;color:#1f77b4'>
🌍 English → Telugu Translator
</h1>
""",
unsafe_allow_html=True
)

st.write("Translate English text into Telugu using a Transformer model.")

sentence = st.text_area(
    "Enter English Sentence",
    height=150
)

if st.button(
    "Translate",
    use_container_width=True
):

    if sentence.strip():

        translation = (
            translator.translate(
                sentence
            )
        )

        st.success(
            translation
        )

    else:

        st.warning(
            "Please enter text."
        )