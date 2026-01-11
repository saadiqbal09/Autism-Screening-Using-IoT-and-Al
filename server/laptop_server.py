from flask import Flask, request, jsonify
import torch
from transformers import pipeline
import tempfile
import os

app = Flask(__name__)

print("Loading Whisper model...")
model = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small",
    device="cuda" if torch.cuda.is_available() else "cpu"
)
print("Model loaded")

@app.route("/analyze", methods=["POST"])
def analyze():
    # Save received data as a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(request.data)
        audio_path = f.name

    try:
        result = model(audio_path)
        text = result["text"]
    except Exception:
        text = "Unable to process audio"

    os.remove(audio_path)

    # Simple decision logic (demo)
    risk = "LOW"
    if len(text.split()) < 3:
        risk = "MODERATE"

    return jsonify({
        "risk": risk,
        "transcript": text
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

