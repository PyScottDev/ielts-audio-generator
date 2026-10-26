"""IELTS Listening Part 2: Gemini 3.8 two-speaker TTS and MP3 assembly.

Uses the existing tests/<TEST_NAME>/text files unchanged.
New 3.8 WAVs and MP3s live in separate raw_38/final_38 folders.
"""

import base64
import os
import random
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

SPEAKER_1 = "KIM"
SPEAKER_2 = "DANIEL"
SPEAKER_1_VOICE = "Kore"
SPEAKER_2_VOICE = "Puck"
NARRATOR_VOICE = "Charon"

# Keep these short. They go into structured speech_metadata, NOT spoken text.
SPEAKER_1_STYLE = (
    "Friendly, practical cycling coach, conversational and relaxed. "
    "Mild, natural Melbourne Australian English. "
    "Measured but natural pace; short pauses between ideas."
)
SPEAKER_2_STYLE = (
    "Relaxed and curious, contemporary British English. "
    "Measured, natural conversational pace."
)
NARRATOR_STYLE = (
    "Restrained British radio continuity announcer. "
    "Clear, composed, natural; measured but not slow."
)

BEGINNING_PAUSE = 33_000
MIDDLE_PAUSE = 26_000
SHORT_PAUSE = 1_000
FINAL_BITRATE = "192k"

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
    from packaging.version import Version
    if Version(version("google-genai")) < Version("2.25.0"):
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


def split_speaker_turns(transcript):
    """Convert 'KIM: ...' / 'DANIEL: ...' lines into structured turns.

    Wrapped continuation lines are kept with the current speaker. Fail loudly
    rather than send a label or direction to the audio model as speech.
    """
    speakers = (SPEAKER_1, SPEAKER_2)
    label = re.compile(r"^\s*(?:\*\*)?([\w -]+)(?:\*\*)?\s*:\s*(.*)$")
    turns = []
    current = None
    lines = []

    def flush():
        if current is not None:
            spoken = "\n".join(lines).strip()
            if not spoken:
                raise ValueError(f"Empty spoken turn for {current}")
            turns.append((current, spoken))

    for original in transcript.splitlines():
        line = original.strip()
        if not line:
            if current is not None and lines and lines[-1] != "":
                lines.append("")
            continue
        match = label.match(line)
        if match:
            possible = match.group(1).strip().upper()
            if possible in speakers:
                flush()
                current = possible
                lines = [match.group(2).strip()]
                continue
            # Reject apparent third-person/stage-direction labels.
            if possible in ("NARRATOR", "PAUSE", "LECTURER", "SPEAKER 1", "SPEAKER 2"):
                raise ValueError(f"Unexpected label in dialogue: {possible}")
        if current is None:
            raise ValueError(
                "Expected dialogue lines beginning with "
                f"'{SPEAKER_1}:' or '{SPEAKER_2}:'. Found: {line[:90]!r}"
            )
        lines.append(line)
    flush()
    if not turns:
        raise ValueError("No dialogue speaker turns found.")
    return turns


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


# def dialogue_request(transcript):
#     styles = {SPEAKER_1: SPEAKER_1_STYLE, SPEAKER_2: SPEAKER_2_STYLE}
#     content = [speech_piece(words, styles[who], who)
#                for who, words in split_speaker_turns(transcript)]
#     return content, {"speakers": [
#         {"speaker": SPEAKER_1, "voice": SPEAKER_1_VOICE},
#         {"speaker": SPEAKER_2, "voice": SPEAKER_2_VOICE},
#     ]}

def dialogue_request(transcript):
    return {
        "type": "user_input",
        "content": [
            {
                "type": "text",
                "text": transcript,
            }
        ],
    }

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

    d1_audio = AudioSegment.from_wav(d1_path)
    d2_audio = AudioSegment.from_wav(d2_path)
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
        path = FINAL_DIR / filename
        audio.export(path, format="mp3", bitrate=FINAL_BITRATE)
        print(f"Created: {path}")
    print("Finished.")


if __name__ == "__main__":
    main()
