"""Generate and assemble an IELTS Listening Part 4 recording."""

import os
import random
import time
import wave
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from pydub import AudioSegment


MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 10
MAX_RETRY_DELAY = 120

# ============================================================
# SETTINGS TO CHANGE FOR EACH TEST
# ============================================================

TEST_NAME = "railway_time_p4"
MODEL = "gemini-3.1-flash-tts-preview"

# Select the underlying Gemini voice for the Part 4 speaker.
SPEAKER_VOICE = "Puck"

PACING_STYLE = """
Use a measured IELTS Listening test pace.
Speak slightly more slowly and deliberately than in an ordinary academic talk.

Do not rush through longer sentences, clauses, lists, contrasts or conclusions.
Allow brief natural pauses at commas and between complete ideas.
Give definitions, contrasts, qualifications and changes of topic enough time to be understood clearly.

Keep the delivery natural, fluent and academically appropriate.
Do not sound sleepy, hesitant, theatrical or unnaturally slow.
Do not stretch individual words.
Create the slower pace through controlled rhythm, clear phrasing and brief pauses.
""".strip()

SPEAKER_STYLE = f"""
Use a clear, natural and internationally understandable English accent.

Sound like a university lecturer or knowledgeable speaker giving an academic-style talk
to an educated general audience.

Keep the delivery clear, engaged and natural rather than conversational or theatrical.
Use subtle emphasis to signal important contrasts, explanations, examples and conclusions.
Do not emphasise words merely because they are answers to the listening questions.

Maintain very clear articulation and a calm, confident delivery.

{PACING_STYLE}
""".strip()


# ============================================================
# SETTINGS THAT CAN REMAIN THE SAME
# ============================================================

NARRATOR_VOICE = "Charon"

NARRATOR_STYLE = """
Speak as a restrained British radio continuity announcer.
Use a clear, internationally intelligible British English accent.
Sound composed, authoritative and natural, without theatrical emphasis.
Use a measured but not slow pace.
""".strip()

# Pause lengths in milliseconds.
# Candidates look at Questions 31-40 together before the monologue begins.
BEGINNING_PAUSE = 33_000

# Brief silent pause at a natural structural boundary in the monologue.
# There is no narrator at this point.
MIDPOINT_PAUSE = 2_000

SHORT_PAUSE = 1_000

# Final MP3 quality.
FINAL_BITRATE = "192k"


# ============================================================
# FOLDERS AND API CLIENT
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
TEST_DIR = BASE_DIR / "tests" / TEST_NAME
TEXT_DIR = TEST_DIR / "text"
RAW_DIR = TEST_DIR / "audio" / "raw"
FINAL_DIR = TEST_DIR / "audio" / "final"

TEXT_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)
FINAL_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError(
        "API key not found. Add GEMINI_API_KEY or GOOGLE_API_KEY "
        "to your .env file."
    )

client = genai.Client(api_key=api_key)


# ============================================================
# VOICE CONFIGURATIONS
# ============================================================

speaker_config = types.SpeechConfig(
    voice_config=types.VoiceConfig(
        prebuilt_voice_config=types.PrebuiltVoiceConfig(
            voice_name=SPEAKER_VOICE
        )
    )
)

narrator_config = types.SpeechConfig(
    voice_config=types.VoiceConfig(
        prebuilt_voice_config=types.PrebuiltVoiceConfig(
            voice_name=NARRATOR_VOICE
        )
    )
)


# ============================================================
# PROMPT CONSTRUCTION
# ============================================================

def create_speaker_prompt(transcript):
    """Combine the speaker directions with one section of the monologue."""

    return f"""
{SPEAKER_STYLE}

Text to speak:

{transcript}
""".strip()


def create_narrator_prompt(narrator_text):
    """Combine the permanent narrator directions with the spoken text."""

    return f"""
{NARRATOR_STYLE}

Text to speak:

{narrator_text}
""".strip()


# ============================================================
# AUDIO GENERATION
# ============================================================

def generate_audio(prompt, wav_filename, speech_config):
    """Generate one WAV file, retrying temporary Gemini failures."""

    output_path = RAW_DIR / wav_filename

    # Preserve successfully generated clips when rerunning the script.
    if output_path.exists() and output_path.stat().st_size > 44:
        print(f"Skipping existing file: {output_path.name}")
        return output_path

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(
                f"Generating {output_path.name} "
                f"(attempt {attempt}/{MAX_RETRIES})..."
            )

            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=speech_config,
                ),
            )

            audio_data = (
                response.candidates[0]
                .content.parts[0]
                .inline_data.data
            )

            # Write to a temporary file first so an interrupted write
            # cannot leave a broken WAV with the final filename.
            temporary_path = output_path.with_suffix(".tmp.wav")

            with wave.open(str(temporary_path), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(24_000)
                wav_file.writeframes(audio_data)

            temporary_path.replace(output_path)

            print(f"Created: {output_path.name}")
            return output_path

        except errors.APIError as error:
            status_code = getattr(error, "code", None)

            # Retry temporary server, timeout and rate-limit errors only.
            retryable = (
                status_code == 408
                or status_code == 429
                or (
                    status_code is not None
                    and 500 <= status_code <= 599
                )
            )

            if not retryable:
                raise

            if attempt == MAX_RETRIES:
                raise RuntimeError(
                    f"Gemini failed to create {output_path.name} "
                    f"after {MAX_RETRIES} attempts."
                ) from error

            base_delay = min(
                INITIAL_RETRY_DELAY * (2 ** (attempt - 1)),
                MAX_RETRY_DELAY,
            )

            delay = base_delay + random.uniform(0, 5)

            print(
                f"Temporary Gemini error {status_code} while generating "
                f"{output_path.name}. Retrying in {delay:.1f} seconds..."
            )

            time.sleep(delay)

    raise RuntimeError(f"Unable to generate {output_path.name}.")


def load_text(filename):
    """Load one UTF-8 text file from the text folder."""

    path = TEXT_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Text file not found: {path}")

    return path.read_text(encoding="utf-8").strip()


# ============================================================
# GENERATE THE TWO MONOLOGUE SECTIONS
# ============================================================

# Split the transcript at the natural structural pause shown by the dotted
# separator in a Cambridge Part 4 transcript. Do not split mechanically
# after a particular question number.
lecture_1_text = load_text("lecture_part_1.txt")
lecture_2_text = load_text("lecture_part_2.txt")

lecture_1_path = generate_audio(
    prompt=create_speaker_prompt(lecture_1_text),
    wav_filename="lecture_part_1.wav",
    speech_config=speaker_config,
)

lecture_2_path = generate_audio(
    prompt=create_speaker_prompt(lecture_2_text),
    wav_filename="lecture_part_2.wav",
    speech_config=speaker_config,
)


# ============================================================
# GENERATE THE NARRATOR ANNOUNCEMENTS
# ============================================================

# Suggested text files:
#
# narrator_intro.txt
#   Part Four. You will hear ...
#   First, you have some time to look at Questions 31 to 40.
#
# narrator_start.txt
#   Now listen carefully and answer Questions 31 to 40.
#
# narrator_end.txt
#   That is the end of Part Four.
#
narrator_files = {
    "intro": "narrator_intro.txt",
    "start": "narrator_start.txt",
    "end": "narrator_end.txt",
}

narrator_paths = {}

for clip_name, text_filename in narrator_files.items():
    narrator_text = load_text(text_filename)

    narrator_paths[clip_name] = generate_audio(
        prompt=create_narrator_prompt(narrator_text),
        wav_filename=f"{clip_name}.wav",
        speech_config=narrator_config,
    )


# ============================================================
# LOAD THE GENERATED WAV FILES
# ============================================================

lecture_1 = AudioSegment.from_wav(lecture_1_path)
lecture_2 = AudioSegment.from_wav(lecture_2_path)

narrator_audio = {
    clip_name: AudioSegment.from_wav(path)
    for clip_name, path in narrator_paths.items()
}


# ============================================================
# CREATE THE PAUSES
# ============================================================

beginning_pause = AudioSegment.silent(
    duration=BEGINNING_PAUSE,
    frame_rate=24_000,
)

midpoint_pause = AudioSegment.silent(
    duration=MIDPOINT_PAUSE,
    frame_rate=24_000,
)

short_pause = AudioSegment.silent(
    duration=SHORT_PAUSE,
    frame_rate=24_000,
)


# ============================================================
# ASSEMBLE THE COMPLETE PART 4
# ============================================================

complete_test = (
    narrator_audio["intro"]
    + beginning_pause
    + narrator_audio["start"]
    + short_pause
    + lecture_1
    + midpoint_pause
    + lecture_2
    + short_pause
    + narrator_audio["end"]
)


# ============================================================
# EXPORT THE FINISHED RECORDING AS MP3
# ============================================================

complete_test_path = FINAL_DIR / f"{TEST_NAME}_complete.mp3"

complete_test.export(
    complete_test_path,
    format="mp3",
    bitrate=FINAL_BITRATE,
)

print()
print("Finished.")
print(f"Created: {complete_test_path}")
