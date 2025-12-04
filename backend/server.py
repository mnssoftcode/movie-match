from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import json

app = FastAPI()

# Load model
model_path = "movie_model_tiny"
model = DistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
model.eval()

# Load genre index map
with open("tag2idx.json", "r") as f:
    tag2idx = json.load(f)
idx2tag = {v: k for k, v in tag2idx.items()}

class MoodInput(BaseModel):
    text: str

@app.post("/predict")
def predict(mood: MoodInput):
    enc = tokenizer(
        mood.text,
        truncation=True,
        padding="max_length",
        max_length=32,
        return_tensors="pt"
    )

    with torch.no_grad():
        logits = model(enc["input_ids"], attention_mask=enc["attention_mask"]).logits
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    ranked = sorted(
        [(idx2tag[i], float(probs[i])) for i in range(len(probs))],
        key=lambda x: x[1],
        reverse=True
    )

    return {"genres": ranked[:5]}

@app.get("/")
def read_root():
    return {"message": "MovieMatch AI Backend is running!"}