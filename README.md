# 🌿 English → Telugu Translator

A Streamlit web app that translates English text to Telugu using the **Helsinki-NLP/opus-mt-en-mul** MarianMT transformer model from Hugging Face.

## 🚀 Live Demo
Deploy instantly on [Streamlit Community Cloud](https://streamlit.io/cloud).

---

## 📁 Project Structure

```
english-telugu-translator/
├── app.py                  # Streamlit UI
├── translate.py            # Translation logic (model load + inference)
├── requirements.txt        # Python dependencies
├── .streamlit/
│   └── config.toml         # Theme & server settings
└── README.md
```

---

## 🛠️ Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/english-telugu-translator.git
cd english-telugu-translator

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

The app opens at **http://localhost:8501**  
> ⚠️ First run downloads the model (~300 MB). Subsequent runs use the cached version.

---

## ☁️ Deploy on Streamlit Community Cloud

1. Push this repo to GitHub (public or private).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Select your repo, branch `main`, and set **Main file path** to `app.py`.
4. Click **Deploy** — done! 🎉

> **Note:** Streamlit Cloud has a 1 GB memory limit on the free tier. The model (~300 MB) fits comfortably.

---

## ⚙️ How It Works

| Step | Detail |
|------|--------|
| Model | `Helsinki-NLP/opus-mt-en-mul` (MarianMT) |
| Target tag | `>>tel<<` prepended to force Telugu output |
| Beam search | 4 beams, no-repeat ngram size 3 |
| Max tokens | 512 input / 512 output |

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit` | Web UI |
| `transformers` | Hugging Face model loading & inference |
| `torch` | PyTorch backend |
| `sentencepiece` | Tokenizer subword model |
| `sacremoses` | Text pre/post-processing |

---

## 📝 License
MIT
