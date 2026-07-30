import json
import os
import random
import time
import wave
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from pydub import AudioSegment, effects


# ============================================================
# SETTINGS
# ============================================================

TEST_NAME = "surname_phone_drill"
MODEL = "gemini-3.1-flash-tts-preview"
FINAL_BITRATE = "320k"

SAMPLE_RATE = 24_000
CHANNELS = 1
SAMPLE_WIDTH = 2

MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 10
MAX_RETRY_DELAY = 120


# ============================================================
# SPEAKERS
# ============================================================

SPEAKERS = {
    "RECEPTIONIST": {
        "voice": "Kore",
        "style": """
Use a clear, internationally understandable Melbourne Australian accent.
This is the same receptionist throughout all ten mini-dialogues.
Sound professional and attentive at first.
Follow the delivery instruction for each turn so that the receptionist's
frustration develops gradually and culminates in a comic meltdown.
Even when frustrated, keep every name, letter and digit intelligible.
""".strip(),
    },
    "CALLER_01": {
        "voice": "Puck",
        "style": """
Use a clear, natural Scottish accent.
Sound polite and matter-of-fact.
Pronounce surname spellings carefully without becoming unnaturally slow.
""".strip(),
    },
    "CALLER_02": {
        "voice": "Aoede",
        "style": """
Use a mild, internationally understandable North American accent.
Sound polite and conversational.
Make the false start and self-correction sound spontaneous.
""".strip(),
    },
    "CALLER_03": {
        "voice": "Fenrir",
        "style": """
Use a light, intelligible Polish accent.
Sound calm, polite and fluent.
Make the contrast between fourteen and forty completely clear.
Do not exaggerate the accent.
""".strip(),
    },
    "CALLER_04": {
        "voice": "Leda",
        "style": """
Use a clear northern English accent.
Sound friendly but precise.
Articulate the individual letters naturally and clearly.
""".strip(),
    },
    "CALLER_05": {
        "voice": "Charon",
        "style": """
Use a natural southern English accent.
Sound relaxed and cooperative.
Speak telephone digits clearly and distinguish fourteen from forty.
""".strip(),
    },
    "CALLER_06": {
        "voice": "Callirrhoe",
        "style": """
Use a clear Canadian accent.
Sound patient and matter-of-fact.
Make the contrast between flat four and number forty easy to hear,
without overemphasising it.
""".strip(),
    },
    "CALLER_07": {
        "voice": "Orus",
        "style": """
Use a mild, internationally understandable New Zealand accent.
Sound friendly and slightly absent-minded.
Make the false start and replacement number sound natural.
""".strip(),
    },
    "CALLER_08": {
        "voice": "Autonoe",
        "style": """
Use a light, intelligible German accent.
Sound calm and cheerful.
Pronounce the surname spellings and final added E clearly.
Do not exaggerate the accent.
""".strip(),
    },
    "CALLER_09": {
        "voice": "Iapetus",
        "style": """
Use a natural, internationally understandable Welsh accent.
Sound careful and helpful, then unexpectedly cheerful about the correction.
Keep every telephone digit distinct.
""".strip(),
    },
    "CALLER_10": {
        "voice": "Despina",
        "style": """
Use a calm, natural southern English accent.
Sound completely innocent and unaware of the receptionist's mounting distress.
Keep fourteen, forty and four clearly distinguishable.
""".strip(),
    },
}


# ============================================================
# DIRECTORIES AND API CLIENT
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
TEST_DIR = PROJECT_DIR / "tests" / TEST_NAME
TEXT_DIR = TEST_DIR / "text"
RAW_DIR = TEST_DIR / "audio" / "raw"
FINAL_DIR = TEST_DIR / "audio" / "final"

RAW_DIR.mkdir(parents=True, exist_ok=True)
FINAL_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(PROJECT_DIR / ".env")

api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise RuntimeError(
        "No Gemini API key was found. Add GEMINI_API_KEY (or GOOGLE_API_KEY) "
        "to the .env file in the project directory."
    )

client = genai.Client(api_key=api_key)


# ============================================================
# LOAD AND VALIDATE THE JSON
# ============================================================

def load_dialogue_turns(filename):
    """Load the dialogue turns and check that the JSON is usable."""

    path = TEXT_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Dialogue file not found: {path}\n"
            f"Place dialogue_turns.json inside {TEXT_DIR}"
        )

    turns = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(turns, list) or not turns:
        raise ValueError("The dialogue JSON must contain a non-empty list.")

    required_fields = {"mini_dialogue", "speaker", "text"}

    for index, turn in enumerate(turns, start=1):
        if not isinstance(turn, dict):
            raise ValueError(f"Turn {index} must be a JSON object.")

        missing_fields = required_fields - turn.keys()

        if missing_fields:
            missing = ", ".join(sorted(missing_fields))
            raise ValueError(f"Turn {index} is missing: {missing}")

        if turn["speaker"] not in SPEAKERS:
            raise ValueError(
                f"Unknown speaker in turn {index}: {turn['speaker']}"
            )

        if not isinstance(turn["text"], str) or not turn["text"].strip():
            raise ValueError(f"Turn {index} has no spoken text.")

        pause_after = turn.get("pause_after", 0)

        if not isinstance(pause_after, int) or pause_after < 0:
            raise ValueError(
                f"Turn {index} has an invalid pause_after value."
            )

        if pause_after:
            is_last_turn = index == len(turns)
            next_dialogue_is_different = (
                not is_last_turn
                and turns[index]["mini_dialogue"] != turn["mini_dialogue"]
            )

            if not is_last_turn and not next_dialogue_is_different:
                raise ValueError(
                    f"Turn {index} inserts a pause inside mini-dialogue "
                    f"{turn['mini_dialogue']}. Pauses should occur only "
                    "after complete mini-dialogues."
                )

    return turns


# ============================================================
# GEMINI SPEECH CONFIGURATION AND PROMPTS
# ============================================================

def create_speaker_config(voice_name):
    """Create a single-speaker Gemini TTS configuration."""

    return types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                voice_name=voice_name
            )
        )
    )


def create_turn_prompt(speaker_name, spoken_text, delivery=""):
    """Create the TTS prompt for one short turn."""

    speaker = SPEAKERS[speaker_name]

    delivery_instruction = (
        delivery.strip()
        if delivery.strip()
        else "Use a natural delivery appropriate to the conversation."
    )

    return f"""
This is one short turn from a sequence of administrative mini-dialogues.

Permanent speaker style:
{speaker["style"]}

Delivery for this turn:
{delivery_instruction}

Speak only the words under "Text to speak".
Do not announce the speaker name.
Do not read any directions aloud.
Do not add, remove, rephrase or correct anything.
Preserve every name, letter and digit exactly as written.
Use a natural conversational pace.

Text to speak:
{spoken_text}
""".strip()


# ============================================================
# WAV FILE HANDLING
# ============================================================

def write_pcm_wav(path, pcm_data):
    """Write Gemini's raw 24 kHz mono PCM data to a WAV file."""

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(CHANNELS)
        wav_file.setsampwidth(SAMPLE_WIDTH)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(pcm_data)


def wav_is_usable(path):
    """Return True when an existing WAV can safely be reused."""

    if not path.exists():
        return False

    try:
        with wave.open(str(path), "rb") as wav_file:
            return (
                wav_file.getnchannels() == CHANNELS
                and wav_file.getsampwidth() == SAMPLE_WIDTH
                and wav_file.getframerate() == SAMPLE_RATE
                and wav_file.getnframes() > 0
            )
    except (wave.Error, EOFError):
        return False


def extract_audio_data(response):
    """Extract the first audio data block from a Gemini response."""

    for candidate in response.candidates or []:
        if not candidate.content:
            continue

        for part in candidate.content.parts or []:
            if part.inline_data and part.inline_data.data:
                return part.inline_data.data

    raise RuntimeError("Gemini returned no audio data.")


# ============================================================
# CREATE PHONE EFFECT
# ============================================================

def apply_phone_effect(audio):
    """Give speech a restrained telephone-line sound."""

    audio = audio.set_channels(1)
    audio = audio.high_pass_filter(300)
    audio = audio.low_pass_filter(3400)

    audio = effects.compress_dynamic_range(
        audio,
        threshold=-22.0,
        ratio=3.0,
        attack=5.0,
        release=50.0,
    )

    return audio.apply_gain(-1.0)


# ============================================================
# GENERATE ONE TURN WITH RETRIES
# ============================================================

def generate_audio(prompt, wav_filename, speech_config):
    """
    Generate one WAV file.

    Existing valid WAVs are reused. Temporary API failures are retried with
    exponential backoff and jitter. A temporary WAV is written first so an
    interrupted run does not leave a partial final clip.
    """

    output_path = RAW_DIR / wav_filename
    temporary_path = RAW_DIR / f"{output_path.stem}.tmp.wav"

    if wav_is_usable(output_path):
        print(f"Reusing:    {output_path.name}")
        return output_path

    if output_path.exists():
        print(f"Replacing unusable WAV: {output_path.name}")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(
                f"Generating: {output_path.name} "
                f"(attempt {attempt}/{MAX_RETRIES})"
            )

            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=speech_config,
                ),
            )

            pcm_data = extract_audio_data(response)
            write_pcm_wav(temporary_path, pcm_data)

            if not wav_is_usable(temporary_path):
                raise RuntimeError(
                    f"Gemini produced an unusable WAV for {wav_filename}."
                )

            temporary_path.replace(output_path)
            return output_path

        except errors.APIError as exc:
            status_code = getattr(exc, "code", None)
            is_temporary = (
                status_code in {408, 429, 500, 502, 503, 504}
                or status_code is None
            )

            if not is_temporary or attempt == MAX_RETRIES:
                raise

            delay = min(
                INITIAL_RETRY_DELAY * (2 ** (attempt - 1)),
                MAX_RETRY_DELAY,
            )
            delay += random.uniform(0, delay * 0.25)

            print(
                f"Temporary API error ({status_code or 'unknown'}). "
                f"Retrying in {delay:.1f} seconds."
            )
            time.sleep(delay)

        finally:
            if temporary_path.exists() and not wav_is_usable(temporary_path):
                temporary_path.unlink()

    raise RuntimeError(f"Could not generate {wav_filename}.")


# ============================================================
# GENERATE AND ASSEMBLE THE COMPLETE DRILL
# ============================================================

def main():
    dialogue_turns = load_dialogue_turns("dialogue_turns.json")
    assembled_dialogue = AudioSegment.empty()

    print(f"Generating {len(dialogue_turns)} dialogue turns...")
    print()

    for index, turn in enumerate(dialogue_turns, start=1):
        speaker_name = turn["speaker"]
        speaker = SPEAKERS[speaker_name]

        safe_speaker_name = speaker_name.lower()
        wav_filename = f"turn_{index:02d}_{safe_speaker_name}.wav"

        turn_path = generate_audio(
            prompt=create_turn_prompt(
                speaker_name=speaker_name,
                spoken_text=turn["text"],
                delivery=turn.get("delivery", ""),
            ),
            wav_filename=wav_filename,
            speech_config=create_speaker_config(speaker["voice"]),
        )

        # assembled_dialogue += AudioSegment.from_wav(turn_path)
        turn_audio = AudioSegment.from_wav(turn_path)

        if speaker_name.startswith("CALLER_"):
            turn_audio = apply_phone_effect(turn_audio)

        assembled_dialogue += turn_audio

        # The JSON supplies pause_after only on the final turn of each
        # mini-dialogue. No artificial silence is inserted between speakers.
        pause_duration = turn.get("pause_after", 0)

        if pause_duration:
            assembled_dialogue += AudioSegment.silent(
                duration=pause_duration,
                frame_rate=SAMPLE_RATE,
            )

    final_path = FINAL_DIR / f"{TEST_NAME}_complete.mp3"

    assembled_dialogue.export(
        final_path,
        format="mp3",
        bitrate=FINAL_BITRATE,
    )

    duration_seconds = len(assembled_dialogue) / 1000
    minutes = int(duration_seconds // 60)
    seconds = duration_seconds % 60

    print()
    print("Finished.")
    print(f"Created:  {final_path}")
    print(f"Duration: {minutes}:{seconds:04.1f}")


if __name__ == "__main__":
    main()