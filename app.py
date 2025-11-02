# app.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydub import AudioSegment
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import numpy as np
import tempfile
import os
hf_token = os.getenv("HF_TOKEN")

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load your Wav2Vec2 model
def load_model():
    processor = Wav2Vec2Processor.from_pretrained("hoanghuy2000gl/Wav2Vec2-VIVOS")
    model = Wav2Vec2ForCTC.from_pretrained("hoanghuy2000gl/Wav2Vec2-VIVOS",
        token=hf_token
    )
    return processor, model


# Initialize model and processor
processor, model = load_model()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)


def process_audio(audio_path):
    try:
        # Load audio
        audio = AudioSegment.from_file(audio_path)

        # Convert to mono and correct sample rate (16kHz for Wav2Vec2)
        audio = audio.set_channels(1).set_frame_rate(16000)

        # Convert to numpy array
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)

        # Normalize
        samples = samples / np.max(np.abs(samples))

        # Process with model
        inputs = processor(samples, sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.batch_decode(predicted_ids)

        return transcription[0]

    except Exception as e:
        print(f"Error processing audio: {e}")
        raise


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model": "Wav2Vec2-VIVOS",
        "device": device
    }


@app.post("/transcribe")
async def transcribe_audio(audio_file: UploadFile = File(...)):
    if not audio_file:
        raise HTTPException(status_code=400, detail="No audio file provided")

    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            # Read uploaded file
            content = await audio_file.read()
            temp_file.write(content)
            temp_path = temp_file.name

        try:
            # Process audio with Wav2Vec2
            text = process_audio(temp_path)

            # Clean up
            os.unlink(temp_path)

            return {
                "success": True,
                "text": text,
                "filename": audio_file.filename
            }

        except Exception as e:
            # Clean up on error
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise HTTPException(
                status_code=500,
                detail=f"Error processing audio: {str(e)}"
            )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error handling file: {str(e)}"
        )


@app.get("/model-info")
async def get_model_info():
    return {
        "model_name": "Wav2Vec2-VIVOS",
        "language": "Vietnamese",
        "input_sample_rate": 16000,
        "input_channels": 1,
        "device": device,
        "supported_formats": ["wav", "mp3", "ogg", "flac"],
    }