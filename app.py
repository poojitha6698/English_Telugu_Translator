import streamlit as st
from translate import translate_text, load_model

st.set_page_config(
    page_title="English → Telugu Translator",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Tiro+Telugu&family=Inter:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.hero {
    background: linear-gradient(135deg, #0f4c35 0%, #1a6b4a 60%, #2d8f63 100%);
    border-radius: 16px;
    padding: 2.5rem 2rem 2rem 2rem;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 4px 24px rgba(15,76,53,0.18);
}
.hero h1 { color: #e8f5ee; font-size: 2rem; font-weight: 600; margin: 0 0 0.4rem 0; letter-spacing: -0.5px; }
.hero .telugu-title { font-family: 'Tiro Telugu', serif; color: #a8dfc0; font-size: 1.25rem; margin: 0; }
.hero .subtitle { color: #c5e8d5; font-size: 0.92rem; margin-top: 0.7rem; }

.box-label { font-size: 0.78rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: #0f4c35; margin-bottom: 0.4rem; }

.output-box {
    background: #f0faf4;
    border: 1.5px solid #a8dfc0;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    min-height: 120px;
    font-family: 'Tiro Telugu', serif;
    font-size: 1.25rem;
    line-height: 1.8;
    color: #0f3d28;
    white-space: pre-wrap;
    word-break: break-word;
}

div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #0f4c35, #1a6b4a);
    color: #e8f5ee;
    border: none;
    border-radius: 10px;
    padding: 0.65rem 2.4rem;
    font-size: 1rem;
    font-weight: 600;
    width: 100%;
    cursor: pointer;
    transition: opacity 0.2s;
}
div[data-testid="stButton"] > button:hover { opacity: 0.88; color: #ffffff; }

.footer { text-align: center; color: #9cbbaa; font-size: 0.78rem; margin-top: 2.5rem; padding-top: 1rem; border-top: 1px solid #e0ede6; }
#MainMenu, footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def get_model():
    return load_model()

st.markdown("""
<div class="hero">
    <h1>🌿 English → Telugu Translator</h1>
    <p class="telugu-title">ఇంగ్లీష్ నుండి తెలుగుకు అనువాదం</p>
    <p class="subtitle">Powered by Helsinki-NLP · Opus-MT Transformer</p>
</div>
""", unsafe_allow_html=True)

with st.spinner("⏳ Loading translation model (first run ~30s)…"):
    tokenizer, model = get_model()

EXAMPLES = [
    "Hello, how are you?",
    "Today is a beautiful day.",
    "I love learning new languages.",
    "What is your name?",
    "The sky is blue.",
]

st.markdown('<p class="box-label">Quick Examples</p>', unsafe_allow_html=True)
cols = st.columns(len(EXAMPLES))
selected_example = None
for i, (col, ex) in enumerate(zip(cols, EXAMPLES)):
    if col.button(f"#{i+1}", key=f"ex_{i}", help=ex):
        selected_example = ex

st.markdown('<p class="box-label" style="margin-top:1.2rem;">English Text</p>', unsafe_allow_html=True)

if "input_text" not in st.session_state:
    st.session_state["input_text"] = ""
if selected_example:
    st.session_state["input_text"] = selected_example

input_text = st.text_area(
    label="",
    value=st.session_state["input_text"],
    height=130,
    placeholder="Type or paste English text here…",
    label_visibility="collapsed",
)

translate_clicked = st.button("🔤 Translate to Telugu")

if translate_clicked:
    text = input_text.strip()
    if not text:
        st.warning("⚠️ Please enter some English text to translate.")
    else:
        with st.spinner("Translating…"):
            result = translate_text(text, tokenizer, model)

        st.markdown('<p class="box-label" style="margin-top:1.4rem;">Telugu Translation</p>', unsafe_allow_html=True)
        st.markdown(f'<div class="output-box">{result["telugu"]}</div>', unsafe_allow_html=True)

        st.markdown("**📋 Copy:**")
        st.code(result["telugu"], language=None)

        with st.expander("ℹ️ Translation Details"):
            st.markdown(f"**Model:** `Helsinki-NLP/opus-mt-en-mul`")
            st.markdown(f"**Input tokens:** {result['input_tokens']}")
            st.markdown(f"**Output tokens:** {result['output_tokens']}")
            st.markdown(f"**Inference time:** {result['time_ms']} ms")

st.markdown("""
<div class="footer">
    Built with ❤️ using Streamlit + Hugging Face Transformers &nbsp;|&nbsp; Helsinki-NLP/opus-mt-en-mul
</div>
""", unsafe_allow_html=True)
