from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from transformers import MarianMTModel, MarianTokenizer
import torch
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="English to Telugu Translator", version="1.0.0")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Model config
MODEL_NAME = "Helsinki-NLP/opus-mt-en-mul"
TELUGU_CODE = ">>tel<<"

tokenizer = None
model = None

def load_model():
    global tokenizer, model
    logger.info("Loading translation model...")
    tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
    model = MarianMTModel.from_pretrained(MODEL_NAME)
    model.eval()
    logger.info("Model loaded successfully!")

@app.on_event("startup")
async def startup_event():
    load_model()

class TranslationRequest(BaseModel):
    text: str

class TranslationResponse(BaseModel):
    original: str
    translated: str
    status: str

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/translate", response_model=TranslationResponse)
async def translate(req: TranslationRequest):
    if not req.text.strip():
        return JSONResponse(status_code=400, content={"error": "Text cannot be empty"})

    try:
        # Prepend Telugu language code
        src_text = f"{TELUGU_CODE} {req.text.strip()}"
        inputs = tokenizer([src_text], return_tensors="pt", padding=True, truncation=True, max_length=512)

        with torch.no_grad():
            translated = model.generate(**inputs, num_beams=4, early_stopping=True)

        result = tokenizer.decode(translated[0], skip_special_tokens=True)

        return TranslationResponse(
            original=req.text.strip(),
            translated=result,
            status="success"
        )
    except Exception as e:
        logger.error(f"Translation error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=False)
