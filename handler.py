"""
RunPod serverless handler for Dia 1.6B TTS.

Accepts a list of dialogue segments, generates audio for each using Dia,
concatenates them into a single MP3, and returns it as base64.

Voice continuity: The last PROMPT_TAIL_SECONDS of each chunk's audio is
passed as an audio_prompt to the next chunk, so Dia generates a consistent
voice across the entire episode instead of picking random voices per chunk.

Input format:
{
    "segments": [
        {"text": "[S1] Hello! [S2] Hi there!", "index": 0},
        {"text": "[S1] How are you? [S2] Great!", "index": 1}
    ],
    "cfg_scale": 3.0,        # optional, default 3.0
    "temperature": 1.3,      # optional, default 1.3
    "top_p": 0.95,           # optional, default 0.95
    "speed_factor": 0.94,    # optional, default 0.94
    "max_new_tokens": 3500,  # optional, default 3072
    "seed": 42               # optional, random if omitted
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

from predict import Predictor

# Load model once at container startup
print("Loading Dia 1.6B model...")
predictor = Predictor()
predictor.setup()
print("Model loaded and ready.")

OUTPUT_SAMPLE_RATE = 44100

# How many seconds of audio tail to use as voice prompt for the next chunk.
# 10s gives Dia enough context to match voice timbre, pacing, and intonation.
PROMPT_TAIL_SECONDS = 10


def extract_tail_text(chunk_text, max_lines=3):
    """Extract the last few lines of a chunk's text for audio_prompt_text.

    Dia uses audio_prompt_text to align the audio prompt with what was said,
    so we give it the transcript of the tail audio we're passing as prompt.
    """
    lines = [l for l in chunk_text.strip().split("\n") if l.strip()]
    return "\n".join(lines[-max_lines:])


def save_audio_prompt(audio_data, sample_rate):
    """Save the tail of an audio chunk as a WAV file for use as audio_prompt."""
    tail_samples = int(PROMPT_TAIL_SECONDS * sample_rate)
    if len(audio_data) <= tail_samples:
        # Chunk is shorter than prompt length — use the whole thing
        tail_audio = audio_data
    else:
        tail_audio = audio_data[-tail_samples:]

    prompt_path = tempfile.mktemp(suffix=".wav")
    sf.write(prompt_path, tail_audio.astype(np.float32), sample_rate)
    return prompt_path


def handler(job):
    """Process a podcast episode generation job."""
    prev_prompt_path = None
    try:
        input_data = job["input"]

        # Support both single text and segmented input
        segments = input_data.get("segments")
        if segments is None:
            text = input_data.get("text", "")
            if not text.strip():
                return {"error": "No text or segments provided"}
            segments = [{"text": text, "index": 0}]

        # Generation parameters
        cfg_scale = float(input_data.get("cfg_scale", 3.0))
        temperature = float(input_data.get("temperature", 1.3))
        top_p = float(input_data.get("top_p", 0.95))
        speed_factor = float(input_data.get("speed_factor", 0.94))
        max_new_tokens = int(input_data.get("max_new_tokens", 3072))
        seed = input_data.get("seed")
        if seed is not None:
            seed = int(seed)

        # Sort segments by index
        segments = sorted(segments, key=lambda s: s.get("index", 0))

        # Generate audio for each segment, chaining voice via audio_prompt
        all_audio = []
        prev_prompt_text = None  # Transcript of the tail audio
        is_first_generation = True  # Track whether we've generated any audio yet

        for i, segment in enumerate(segments):
            seg_text = segment["text"]
            if not seg_text.strip():
                continue

            print(f"Generating segment {i + 1}/{len(segments)}: {len(seg_text)} chars")

            # First generation: use seed to establish consistent voices.
            # Subsequent: voice is anchored by the audio_prompt chain.
            seg_seed = seed if (seed is not None and is_first_generation) else None

            # Build audio_prompt args for voice continuity
            audio_prompt_arg = None
            audio_prompt_text_arg = None
            prompt_seconds = PROMPT_TAIL_SECONDS

            if prev_prompt_path is not None:
                audio_prompt_arg = prev_prompt_path
                audio_prompt_text_arg = prev_prompt_text
                print(f"  Using {PROMPT_TAIL_SECONDS}s audio prompt from previous chunk")

            output_path = predictor.predict(
                text=seg_text,
                audio_prompt=audio_prompt_arg,
                audio_prompt_text=audio_prompt_text_arg,
                max_new_tokens=max_new_tokens,
                max_audio_prompt_seconds=prompt_seconds,
                cfg_scale=cfg_scale,
                temperature=temperature,
                top_p=top_p,
                cfg_filter_top_k=45,
                speed_factor=speed_factor,
                seed=seg_seed,
            )

            # Read the WAV output as numpy
            audio_data, sr = sf.read(str(output_path))
            all_audio.append(audio_data)
            is_first_generation = False

            # Clean up previous audio prompt temp file
            if prev_prompt_path is not None:
                try:
                    os.unlink(prev_prompt_path)
                except OSError:
                    pass

            # Save this chunk's tail as the audio prompt for the next chunk
            if i < len(segments) - 1:
                prev_prompt_path = save_audio_prompt(audio_data, sr)
                prev_prompt_text = extract_tail_text(seg_text)
            else:
                prev_prompt_path = None
                prev_prompt_text = None

            # Clean up the output WAV
            try:
                os.unlink(str(output_path))
            except OSError:
                pass

            # Add a short pause between segments (0.3s silence)
            if i < len(segments) - 1:
                pause = np.zeros(int(OUTPUT_SAMPLE_RATE * 0.3))
                all_audio.append(pause)

            print(f"  Segment {i + 1} done: {len(audio_data) / sr:.1f}s")

        if not all_audio:
            return {"error": "No audio generated from any segment"}

        # Concatenate all audio
        full_audio = np.concatenate(all_audio)
        duration_seconds = len(full_audio) / OUTPUT_SAMPLE_RATE
        print(f"Total audio: {duration_seconds:.1f}s from {len(segments)} segments")

        # Write concatenated WAV to temp file
        wav_path = tempfile.mktemp(suffix=".wav")
        sf.write(wav_path, full_audio, OUTPUT_SAMPLE_RATE)

        # Convert to MP3 using ffmpeg (64kbps mono — small for base64 transfer)
        mp3_path = wav_path.replace(".wav", ".mp3")
        output_format = "mp3"

        try:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-i", wav_path,
                    "-codec:a", "libmp3lame",
                    "-b:a", "64k",
                    "-ar", "44100",
                    "-ac", "1",
                    mp3_path,
                ],
                capture_output=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg error: {e.stderr.decode()}")
            # Fall back to WAV
            mp3_path = wav_path
            output_format = "wav"

        # Read and encode as base64
        with open(mp3_path, "rb") as f:
            audio_bytes = f.read()

        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

        # Cleanup
        for path in [wav_path, mp3_path]:
            try:
                os.unlink(path)
            except OSError:
                pass

        print(
            f"Done: {len(segments)} segments, "
            f"{duration_seconds:.1f}s, "
            f"{len(audio_bytes):,} bytes {output_format}"
        )

        return {
            "audio_base64": audio_b64,
            "duration_seconds": round(duration_seconds, 2),
            "format": output_format,
            "segment_count": len(segments),
            "file_size_bytes": len(audio_bytes),
        }

    except Exception as e:
        # Clean up any lingering audio prompt temp file
        if prev_prompt_path is not None:
            try:
                os.unlink(prev_prompt_path)
            except OSError:
                pass
        print(f"Handler error: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
