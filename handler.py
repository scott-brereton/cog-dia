"""
RunPod serverless handler for Dia 1.6B TTS.

Accepts a list of dialogue segments, generates audio for each using Dia,
concatenates them into a single MP3, and returns it as base64.

Input format:
{
    "segments": [
        {"text": "[S1] Hello! [S2] Hi there!", "index": 0},
        {"text": "[S1] How are you? [S2] Great!", "index": 1}
    ],
    "cfg_scale": 3.0,        # optional, default 3.0
    "temperature": 1.3,      # optional, default 1.3
    "top_p": 0.95,            # optional, default 0.95
    "speed_factor": 0.94,     # optional, default 0.94
    "max_new_tokens": 3500,   # optional, default 3500
    "seed": 42                # optional, random if omitted
}

Output format:
{
    "audio_base64": "<base64 MP3>",
    "duration_seconds": 720.5,
    "format": "mp3",
    "segment_count": 25,
    "file_size_bytes": 5760000
}
"""

import runpod
import base64
import os
import subprocess
import tempfile
import numpy as np
import soundfile as sf

# Patch predict.py to use our pre-downloaded model weights
os.environ.setdefault("DIA_MODEL_DIR", "/app/model_cache/nari-labs/Dia-1.6B")

from predict import Predictor

# Load model once at startup (runs during container init, not per-request)
print("Loading Dia 1.6B model...")
predictor = Predictor()
predictor.setup()
print("Model loaded and ready.")


def handler(job):
    """Process a podcast episode generation job."""
    input_data = job["input"]

    # Support both single text and segmented input
    segments = input_data.get("segments")
    if segments is None:
        # Single text input (backward compat)
        text = input_data.get("text", "")
        segments = [{"text": text, "index": 0}]

    # Generation parameters
    cfg_scale = input_data.get("cfg_scale", 3.0)
    temperature = input_data.get("temperature", 1.3)
    top_p = input_data.get("top_p", 0.95)
    speed_factor = input_data.get("speed_factor", 0.94)
    max_new_tokens = input_data.get("max_new_tokens", 3500)
    seed = input_data.get("seed")

    # Sort segments by index
    segments = sorted(segments, key=lambda s: s.get("index", 0))

    # Generate audio for each segment
    all_audio = []
    sample_rate = 44100

    for i, segment in enumerate(segments):
        seg_text = segment["text"]
        if not seg_text.strip():
            continue

        print(f"Generating segment {i + 1}/{len(segments)}: {len(seg_text)} chars")

        try:
            output_path = predictor.predict(
                text=seg_text,
                max_new_tokens=max_new_tokens,
                cfg_scale=cfg_scale,
                temperature=temperature,
                top_p=top_p,
                speed_factor=speed_factor,
                seed=seed + i if seed is not None else None,
            )

            # Read the WAV output
            audio_data, sr = sf.read(str(output_path))
            sample_rate = sr
            all_audio.append(audio_data)

            # Add a short pause between segments (0.3s silence)
            if i < len(segments) - 1:
                pause = np.zeros(int(sr * 0.3))
                all_audio.append(pause)

        except Exception as e:
            print(f"Error generating segment {i}: {e}")
            return {"error": f"Failed on segment {i}: {str(e)}"}

    if not all_audio:
        return {"error": "No audio generated"}

    # Concatenate all audio segments
    full_audio = np.concatenate(all_audio)
    duration_seconds = len(full_audio) / sample_rate

    # Write concatenated WAV to temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_file:
        wav_path = wav_file.name
        sf.write(wav_path, full_audio, sample_rate)

    # Convert to MP3 using ffmpeg (much smaller for base64 transfer)
    mp3_path = wav_path.replace(".wav", ".mp3")
    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", wav_path,
                "-codec:a", "libmp3lame",
                "-b:a", "64k",       # 64kbps — good for speech, small files
                "-ar", "44100",
                "-ac", "1",           # mono — speech doesn't need stereo
                mp3_path,
            ],
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e.stderr.decode()}")
        # Fall back to WAV if ffmpeg fails
        mp3_path = wav_path

    # Read and encode as base64
    with open(mp3_path, "rb") as f:
        audio_bytes = f.read()

    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

    # Cleanup temp files
    for path in [wav_path, mp3_path]:
        try:
            os.unlink(path)
        except OSError:
            pass

    output_format = "mp3" if mp3_path.endswith(".mp3") else "wav"

    print(
        f"Generated {len(segments)} segments, "
        f"{duration_seconds:.1f}s total, "
        f"{len(audio_bytes)} bytes {output_format}"
    )

    return {
        "audio_base64": audio_b64,
        "duration_seconds": round(duration_seconds, 2),
        "format": output_format,
        "segment_count": len(segments),
        "file_size_bytes": len(audio_bytes),
    }


runpod.serverless.start({"handler": handler})
