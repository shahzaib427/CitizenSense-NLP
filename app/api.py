"""FastAPI demo.  Run from the project root:  uvicorn app.api:app --reload
Docs UI: http://127.0.0.1:8000/docs"""
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from app.utils import analyze_feedback  # noqa: E402

app = FastAPI(title="CitizenSense API", version="1.0",
              description="Prototype NLP service for public-feedback analysis. Predictions are suggestions only.")


class Feedback(BaseModel):
    text: str = Field(..., min_length=3, examples=["There is no water supply in our area for three days."])


class Batch(BaseModel):
    texts: List[str]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze-feedback")
def analyze(item: Feedback):
    result = analyze_feedback(item.text)
    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])
    return result


@app.post("/analyze-batch")
def analyze_batch(batch: Batch):
    return [analyze_feedback(t) for t in batch.texts[:200]]
