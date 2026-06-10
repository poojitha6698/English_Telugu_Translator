# 🌐 English → Telugu Translator

A full-stack AI translation app using **FastAPI** + **Hugging Face MarianMT Transformer**.

---

## ✨ Features
- Real-time English to Telugu translation
- Helsinki-NLP `opus-mt-en-mul` transformer model
- Clean, modern UI with example phrases
- Copy-to-clipboard support
- REST API with `/translate` endpoint
- Docker-ready for cloud deployment

---

## 📁 Project Structure

```
english-telugu-translator/
├── app.py                 # FastAPI backend
├── requirements.txt
├── Procfile               # For Render/Railway
├── Dockerfile             # For Docker deployment
├── templates/
│   └── index.html         # Frontend UI
└── static/
    ├── css/style.css
    └── js/main.js
```

---

## 🚀 Local Setup

```bash
# 1. Clone & enter project
git clone <your-repo-url>
cd english-telugu-translator

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the server
uvicorn app:app --reload --port 8000

# 5. Open in browser
# http://localhost:8000
```

> ⚠️ First run downloads the model (~300 MB). Subsequent runs are instant.

---

## 🌍 Deploy to Render (Free)

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → New Web Service
3. Connect your GitHub repo
4. Set:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn app:app --host 0.0.0.0 --port $PORT`
5. Click **Deploy**

---

## 🐳 Deploy with Docker

```bash
docker build -t en-te-translator .
docker run -p 8000:8000 en-te-translator
```

---

## 🔌 API Reference

### `POST /translate`
```json
// Request
{ "text": "Good morning" }

// Response
{
  "original": "Good morning",
  "translated": "శుభోదయం",
  "status": "success"
}
```

### `GET /health`
```json
{ "status": "ok", "model_loaded": true }
```

---

## 🧠 Model Info

| Property | Value |
|----------|-------|
| Model | `Helsinki-NLP/opus-mt-en-mul` |
| Architecture | MarianMT (Transformer) |
| Target Language Code | `>>tel<<` |
| Source | [Hugging Face](https://huggingface.co/Helsinki-NLP/opus-mt-en-mul) |

---

Built with ♥ using FastAPI + Hugging Face Transformers
