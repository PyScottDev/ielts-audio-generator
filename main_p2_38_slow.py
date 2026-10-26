"""IELTS Listening Part 2: Gemini 3.8 monologue TTS and MP3 assembly.

Uses the existing tests/<TEST_NAME>/text files unchanged.
New 3.8 WAVs and MP3s live in separate raw_38/final_38 folders.
"""

import base64
import os
import random
import shutil
import subprocess
import re
import time
import wave
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError

from dotenv import load_dotenv
from google import genai
from google.genai import errors
from pydub import AudioSegment

# ----------------------- PER-TEST SETTINGS -----------------------
TEST_NAME = "2_cycle_club_38"
MODEL = "gemini-3.8-flash-tts"

SPEAKER = "KIM"
SPEAKER_VOICE = "Kore"
NARRATOR_VOICE = "Charon"

# Keep these short. They go into structured speech_metadata, NOT spoken text.
SPEAKER_STYLE = (
    "Friendly, practical cycling coach, conversational and relaxed. "
    "Mild, natural Melbourne Australian English. "
    "Measured but natural pace; short pauses between ideas."
)
NARRATOR_STYLE = (
    "Restrained British radio continuity announcer. "
    "Clear, composed, natural; measured but not slow."
)

BEGINNING_PAUSE = 33_000
MIDDLE_PAUSE = 26_000
SHORT_PAUSE = 1_000
FINAL_BITRATE = "192k"

# Pitch-preserving time stretch for KIM ONLY; 1.0 = no change.
# Start with 0.95 (5% slower). Try 0.93 if still too fast.
# This does NOT modify raw WAV files or timed/narrator pauses.
KIM_SPEED = 0.95

MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 10
MAX_RETRY_DELAY = 120

# -------------------------- PATHS -------------------------------
BASE_DIR = Path(__file__).resolve().parent
TEST_DIR = BASE_DIR / "tests" / TEST_NAME
TEXT_DIR = TEST_DIR / "text"
RAW_DIR = TEST_DIR / "audio" / "raw_38"
FINAL_DIR = TEST_DIR / "audio" / "final_38"
RAW_DIR.mkdir(parents=True, exist_ok=True)
FINAL_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("Set GEMINI_API_KEY or GOOGLE_API_KEY in your .env file.")

try:
    installed_sdk = tuple(int(x) for x in version("google-genai").split(".")[:3])
    if installed_sdk < (2, 25, 0):
        raise RuntimeError("Upgrade first: python -m pip install -U 'google-genai>=2.25.0'")
except PackageNotFoundError as exc:
    raise RuntimeError("Install google-genai>=2.25.0") from exc

client = genai.Client(api_key=api_key)


def load_text(filename):
    path = TEXT_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(f"Missing text file: {path}")
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        raise ValueError(f"Empty text file: {path}")
    return text


def monologue_text(source):
    """Accept plain monologue text, optionally prefixed by KIM:.

    The seven prepared text files contain no speaker labels.
    """
    source = re.sub(r"^\s*(?:\*\*)?KIM(?:\*\*)?\s*:\s*", "", source, flags=re.I)
    if re.search(r"(?im)^\s*(?:NARRATOR|DANIEL)\s*:", source):
        raise ValueError("The Kim monologue text contains another speaker label.")
    if re.search(r"\[(?:pause|silence|music)\]", source, re.I):
        raise ValueError("Monologue contains bracketed stage directions.")
    return source.strip()


def narrator_spoken_text(source):
    """Strip only a leading narrator label; do not send cues as speech."""
    source = re.sub(r"^\s*(?:\*\*)?NARRATOR(?:\*\*)?\s*:\s*", "", source, flags=re.I)
    if re.search(r"\[(?:pause|silence|music)\]", source, re.I):
        raise ValueError("Narrator text contains bracketed stage directions. Remove them first.")
    return source.strip()


def speech_piece(words, style, speaker=None):
    metadata = {"type": "speech_metadata", "style": style}
    if speaker is not None:
        metadata["speaker"] = speaker
    return {"type": "text", "text": words, "annotations": [metadata]}


def dialogue_request(transcript):
    """Create a SINGLE-SPEAKER request for Kim's Part 2 monologue."""
    content = [speech_piece(monologue_text(transcript), SPEAKER_STYLE)]
    return content, [{"voice": SPEAKER_VOICE}]


def narrator_request(source):
    return [speech_piece(narrator_spoken_text(source), NARRATOR_STYLE)], [
        {"voice": NARRATOR_VOICE}
    ]


def valid_wav(path):
    if not path.is_file() or path.stat().st_size <= 44:
        return False
    try:
        with wave.open(str(path), "rb") as audio:
            return audio.getnframes() > 0 and audio.getframerate() > 0
    except (OSError, EOFError, wave.Error):
        return False


def generate_audio(content, speech_config, wav_filename):
    output_path = RAW_DIR / wav_filename
    if valid_wav(output_path):
        print(f"Reusing: {output_path.name}")
        return output_path

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"Generating {wav_filename} (attempt {attempt}/{MAX_RETRIES})...")
            response = client.interactions.create(
                model=MODEL,
                input=[{"type": "user_input", "content": content}],
                response_format={"type": "audio"},
                generation_config={"speech_config": speech_config},
            )
            if response.output_audio is None or not response.output_audio.data:
                raise RuntimeError(f"No output audio returned for {wav_filename}.")
            wav_bytes = base64.b64decode(response.output_audio.data)
            if not wav_bytes.startswith(b"RIFF") or wav_bytes[8:12] != b"WAVE":
                raise RuntimeError("Expected WAV output with RIFF header; not saving unexpected audio.")
            temporary_path = output_path.with_suffix(".tmp.wav")
            temporary_path.write_bytes(wav_bytes)  # 3.8 returns a COMPLETE WAV
            if not valid_wav(temporary_path):
                temporary_path.unlink(missing_ok=True)
                raise RuntimeError(f"Generated invalid WAV for {wav_filename}.")
            temporary_path.replace(output_path)
            print(f"Created: {output_path.name}")
            return output_path
        except errors.APIError as error:
            status = getattr(error, "code", None)
            if status not in (408, 429) and not (isinstance(status, int) and 500 <= status <= 599):
                raise
            if attempt == MAX_RETRIES:
                raise RuntimeError(f"API error after {MAX_RETRIES} attempts: {wav_filename}") from error
            delay = min(INITIAL_RETRY_DELAY * 2 ** (attempt - 1), MAX_RETRY_DELAY)
            delay += random.uniform(0, 5)
            print(f"Temporary API error {status}. Retrying in {delay:.1f}s...")
            time.sleep(delay)
    raise RuntimeError(f"Unable to create {wav_filename}")


def adjusted_kim_audio(wav_path):
    """Time-stretch only Kim's audio using FFmpeg atempo (pitch preserved).

    The original WAV is never modified. Silence and narrator clips are not processed.
    """
    if not 0.5 <= KIM_SPEED <= 2.0:
        raise ValueError("KIM_SPEED must be between 0.5 and 2.0")
    if KIM_SPEED == 1.0:
        return AudioSegment.from_wav(wav_path)

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg is required for KIM_SPEED; install ffmpeg or set KIM_SPEED=1.0")

    # Bytes are loaded from the WAV path. FFmpeg streams adjusted WAV back to Python;
    # no intermediate files or extra API calls required.
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
        "-i", str(wav_path), "-vn", "-af", f"atempo={KIM_SPEED}",
        "-f", "wav", "pipe:1",
    ]
    import io
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed on {wav_path.name}: {result.stderr.decode(errors='replace')}")
    adjusted = AudioSegment.from_file(io.BytesIO(result.stdout), format="wav")
    print(f"Adjusted {wav_path.name}: {KIM_SPEED:.2f}x; "
          f"{AudioSegment.from_wav(wav_path).duration_seconds:.1f}s -> "
          f"{adjusted.duration_seconds:.1f}s")
    return adjusted


def main():
    # Fail before making billable calls if input files/labels are incorrect.
    d1 = load_text("dialogue_q1_5.txt")
    d2 = load_text("dialogue_q6_10.txt")
    narrator_files = {
        "intro": "narrator_intro.txt",
        "start_1": "narrator_start_q1_5.txt",
        "middle": "narrator_middle.txt",
        "start_2": "narrator_start_q6_10.txt",
        "end": "narrator_end.txt",
    }
    d1_request = dialogue_request(d1)
    d2_request = dialogue_request(d2)
    narrators = {name: narrator_request(load_text(filename))
                 for name, filename in narrator_files.items()}

    d1_path = generate_audio(*d1_request, "dialogue_q1_5.wav")
    d2_path = generate_audio(*d2_request, "dialogue_q6_10.wav")
    narrator_paths = {name: generate_audio(*request, f"{name}.wav")
                      for name, request in narrators.items()}

    d1_audio = adjusted_kim_audio(d1_path)
    d2_audio = adjusted_kim_audio(d2_path)
    n = {name: AudioSegment.from_wav(path) for name, path in narrator_paths.items()}
    pause_begin = AudioSegment.silent(duration=BEGINNING_PAUSE, frame_rate=24_000)
    pause_middle = AudioSegment.silent(duration=MIDDLE_PAUSE, frame_rate=24_000)
    pause_short = AudioSegment.silent(duration=SHORT_PAUSE, frame_rate=24_000)

    part_one = n["intro"] + pause_begin + n["start_1"] + pause_short + d1_audio
    part_two = (n["middle"] + pause_middle + n["start_2"] + pause_short
                + d2_audio + pause_short + n["end"])
    complete = part_one + pause_short + part_two

    outputs = {
        f"{TEST_NAME}_questions_1-5.mp3": part_one,
        f"{TEST_NAME}_questions_6-10.mp3": part_two,
        f"{TEST_NAME}_complete.mp3": complete,
    }
    for filename, audio in outputs.items():
        # Keep the original MP3s intact; each KIM_SPEED gets distinct filenames.
        stem, extension = filename.rsplit(".", 1)
        speed_tag = str(round(KIM_SPEED * 100)).zfill(3)
        path = FINAL_DIR / f"{stem}_speed{speed_tag}.{extension}"
        audio.export(path, format="mp3", bitrate=FINAL_BITRATE)
        print(f"Created: {path}")
    print("Finished.")


if __name__ == "__main__":
    main()
