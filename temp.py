# app.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydub import AudioSegment
import io
import os
import tempfile

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/transcribe")
async def transcribe_audio(audio_file: UploadFile = File(...)):
    import random

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
            # Load and process audio
            audio = AudioSegment.from_file(temp_path)
            audio = audio.set_channels(1).set_frame_rate(16000)

            # For testing purposes
            random_texts = [
                "Alo 1234, có ai ở nhà không ?",
                "Xin chào, xin chào, tôi là người Việt Nam"
            ]
            text = random.choice(random_texts)

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

# Optional: Add endpoint to get supported file types
@app.get("/supported-formats")
async def get_supported_formats():
    return {
        "formats": [
            "wav",
            "mp3",
            "ogg",
            "flac"
        ]
    }