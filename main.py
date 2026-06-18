from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Annotated

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification

from ai_analysis import generate_ai_analysis

# ==================================================
# Model Configuration
# ==================================================

MODEL_PATH = "fake_review_model"

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(
        f"Model directory not found: {MODEL_PATH}"
    )

# ==================================================
# Device Selection
# ==================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using device: {device}")

# ==================================================
# Load Model
# ==================================================

tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)

model = BertForSequenceClassification.from_pretrained(
    MODEL_PATH
)

model.to(device)
model.eval()

# ==================================================
# Label Mapping
# ==================================================

id2label = {
    0: "OR",  # Original Review
    1: "CG"   # Computer Generated / Fake
}

# ==================================================
# FastAPI App
# ==================================================

app = FastAPI(
    title="Fake Review Detection API",
    version="1.0.0"
)

# ==================================================
# CORS
# ==================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change for production
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================================================
# Static Files
# ==================================================

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)

# ==================================================
# Routes
# ==================================================

@app.get("/")
def serve_ui():
    return FileResponse("static/index.html")


@app.get("/status")
def status():
    return {
        "status": "running",
        "device": str(device),
        "model": "BERT Fake Review Detector"
    }

# ==================================================
# Request Schema
# ==================================================

class ReviewInput(BaseModel):
    message: Annotated[
        str,
        Field(
            min_length=5,
            max_length=5000,
            description="Review text"
        )
    ]

# ==================================================
# Prediction Endpoint
# ==================================================

@app.post("/predict/")
def predict_review(data: ReviewInput):

    try:

        review_text = data.message.strip()

        if not review_text:
            raise HTTPException(
                status_code=400,
                detail="Review cannot be empty."
            )

        # Tokenize
        inputs = tokenizer(
            review_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        # Predict
        with torch.no_grad():
            outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )

        predicted_class = torch.argmax(
            probabilities,
            dim=1
        ).item()

        confidence = probabilities[
            0,
            predicted_class
        ].item()

        confidence_percent = round(
            confidence * 100,
            2
        )

        label = id2label.get(
            predicted_class,
            "UNKNOWN"
        )

        # Generate AI explanation
        try:
            display_prediction = (
            "Genuine Review"
            if label == "OR"
            else "Deceptive Review"
)

            ai_analysis = generate_ai_analysis(
                review=review_text,
                prediction=display_prediction,
                confidence=confidence_percent
            )

        except Exception as ai_error:

            print(
                f"AI Analysis Error: {ai_error}"
            )

            ai_analysis = (
                "AI analysis is currently unavailable."
            )

        return {
            "review": review_text,
            "label": label,
            "class_id": predicted_class,
            "confidence": confidence_percent,
            "ai_analysis": ai_analysis
        }

    except HTTPException:
        raise

    except Exception as e:

        print(f"Prediction Error: {e}")

        raise HTTPException(
            status_code=500,
            detail="Internal server error."
        )

