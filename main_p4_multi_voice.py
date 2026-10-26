"""Generate and assemble an IELTS Listening Part 1 recording."""

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
MODEL = "gemini-2.5-pro-preview-tts"
#MODEL = "gemini-3.1-flash-tts-preview"

# Speaker names must match the labels used in the transcript files.
SPEAKER_1 = "LECTURER"

# These select the underlying Gemini voices.
SPEAKER_1_VOICE = "Puck"

PACING_STYLE = """
Use a measured IELTS Listening test pace.
Speak slightly more slowly and deliberately than in an ordinary casual conversation.

Do not rush through longer sentences, clauses, lists, contrasts or the ends of speaking turns.
Allow brief natural pauses at commas and between complete ideas.

Give contrasts, corrections and conclusions enough time to be understood clearly.

Keep the delivery natural and engaged.
Do not sound sleepy, hesitant, theatrical or unnaturally slow.
Do not stretch individual words.
Create the slower pace through controlled rhythm, clear phrasing and brief pauses.
""".strip()

# Change these independently for each test.
SPEAKER_1_VOICE = "Kore"

SPEAKER_1_STYLE = f"""
Use a clear, natural and internationally understandable British English accent.

Sound like a university lecturer in their thirties or forties giving an academic talk to students.
Be knowledgeable, calm and engaged, without sounding overly formal, theatrical or like a newsreader.

Keep the delivery natural and continuous, as though explaining a familiar subject from prepared notes.
Use subtle emphasis for important contrasts, explanations and transitions between ideas, but do not emphasise information merely because it may be an answer.

Maintain a consistent voice, accent, pitch and speaking style throughout.
Use clear articulation without sounding unnaturally careful.

{PACING_STYLE}
""".strip()



# ============================================================
# SETTINGS THAT CAN REMAIN THE SAME
# ============================================================

NARRATOR_VOICE = "Charon"

CONVERSATION_STYLE = """

Perform this as a natural university lecture delivered to a small group of students.

Sound engaged and conversational rather than formal or announcer-like.

Keep the delivery clear and controlled, but allow normal variation in rhythm and emphasis.

Use subtle emphasis for contrasts and transitions, without sounding theatrical.

Give contrasts, corrections and conclusions enough time to be clear.

Use natural phrasing and connected speech, with brief pauses between complete ideas and sections.

Do not treat paragraph breaks as new speaking turns.

Do not introduce any additional speakers, vocalisations or sounds.

Do not emphasise information merely because it may be an answer.

""".strip()

# CONVERSATION_STYLE = """
# Perform this as a continuous academic monologue for an IELTS Listening Part 4 recording.

# The lecturer should sound natural, composed and engaged, as if giving a prepared talk to students.
# Maintain a steady speaking style, voice, accent, pitch and volume throughout.
# Use natural phrasing and connected speech, with brief pauses between complete ideas and sections.

# Do not treat paragraph breaks as new speaking turns.
# Do not introduce any additional speakers, vocalisations or sounds.
# Do not emphasise information merely because it may be an answer.
# """.strip()

NARRATOR_STYLE = """
Speak as a restrained British radio continuity announcer.
Use a clear, internationally intelligible British English accent.
Sound composed, authoritative and natural, without theatrical emphasis.
Use a measured but not slow pace.
""".strip()

# Pause lengths in milliseconds.
BEGINNING_PAUSE = 33_000
MIDDLE_PAUSE = 0
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

dialogue_config = types.SpeechConfig(
    voice_config=types.VoiceConfig(
        prebuilt_voice_config=types.PrebuiltVoiceConfig(
            voice_name=SPEAKER_1_VOICE
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

def create_dialogue_prompt(transcript):
    """Combine the conversation and speaker-specific directions."""

    return f"""
{CONVERSATION_STYLE}

Individual speaker directions:

{SPEAKER_1}:
{SPEAKER_1_STYLE}



Transcript:

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
                or status_code is not None
                and 500 <= status_code <= 599
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
# GENERATE THE TWO DIALOGUE SECTIONS
# ============================================================

dialogue_1_text = load_text("dialogue_q1_5.txt")
dialogue_2_text = load_text("dialogue_q6_10.txt")

dialogue_1_path = generate_audio(
    prompt=create_dialogue_prompt(dialogue_1_text),
    wav_filename="dialogue_q1_5.wav",
    speech_config=dialogue_config,
)

dialogue_2_path = generate_audio(
    prompt=create_dialogue_prompt(dialogue_2_text),
    wav_filename="dialogue_q6_10.wav",
    speech_config=dialogue_config,
)


# ============================================================
# GENERATE THE NARRATOR ANNOUNCEMENTS
# ============================================================

narrator_files = {
    "intro": "narrator_intro.txt",
    "start_1": "narrator_start_q1_5.txt",
    "middle": "narrator_middle.txt",
    "start_2": "narrator_start_q6_10.txt",
    "end": "narrator_end.txt",
}

narrator_paths = {}

for clip_name, text_filename in narrator_files.items():
    narrator_text = load_text(text_filename)

    if not narrator_text:
        narrator_paths[clip_name] = None
        print(f"Skipping empty narrator file: {text_filename}")
        continue

    narrator_paths[clip_name] = generate_audio(
        prompt=create_narrator_prompt(narrator_text),
        wav_filename=f"{clip_name}.wav",
        speech_config=narrator_config,
    )


# ============================================================
# LOAD THE GENERATED WAV FILES
# ============================================================

dialogue_1 = AudioSegment.from_wav(dialogue_1_path)
dialogue_2 = AudioSegment.from_wav(dialogue_2_path)

print(f"Dialogue 1 overall: {dialogue_1.dBFS:.1f} dBFS")
print(f"Dialogue 1 first 10s: {dialogue_1[:10000].dBFS:.1f} dBFS")
print(f"Dialogue 1 last 10s:  {dialogue_1[-10000:].dBFS:.1f} dBFS")

print(f"Dialogue 2 overall: {dialogue_2.dBFS:.1f} dBFS")
print(f"Dialogue 2 first 10s: {dialogue_2[:10000].dBFS:.1f} dBFS")
print(f"Dialogue 2 last 10s:  {dialogue_2[-10000:].dBFS:.1f} dBFS")

narrator_audio = {
    clip_name: (
        AudioSegment.from_wav(path)
        if path is not None
        else AudioSegment.silent(duration=0, frame_rate=24_000)
    )
    for clip_name, path in narrator_paths.items()
}


# ============================================================
# CREATE THE PAUSES
# ============================================================

pause_33 = AudioSegment.silent(
    duration=BEGINNING_PAUSE,
    frame_rate=24_000,
)

pause_26 = AudioSegment.silent(
    duration=MIDDLE_PAUSE,
    frame_rate=24_000,
)

short_pause = AudioSegment.silent(
    duration=SHORT_PAUSE,
    frame_rate=24_000,
)


# ============================================================
# ASSEMBLE QUESTIONS 1–5
# ============================================================

questions_1_5 = (
    narrator_audio["intro"]
    + pause_33
    + narrator_audio["start_1"]
    + short_pause
    + dialogue_1
)


# ============================================================
# ASSEMBLE QUESTIONS 6–10
# ============================================================

questions_6_10 = (
    narrator_audio["middle"]
    + pause_26
    + narrator_audio["start_2"]
    + short_pause
    + dialogue_2
    + short_pause
    + narrator_audio["end"]
)


# ============================================================
# ASSEMBLE THE COMPLETE TEST
# ============================================================

complete_test = (
    questions_1_5
    + short_pause
    + questions_6_10
)


# ============================================================
# EXPORT THE FINISHED RECORDINGS AS MP3
# ============================================================

questions_1_5_path = (
    FINAL_DIR / f"{TEST_NAME}_questions_1-5.mp3"
)

questions_6_10_path = (
    FINAL_DIR / f"{TEST_NAME}_questions_6-10.mp3"
)

complete_test_path = (
    FINAL_DIR / f"{TEST_NAME}_complete.mp3"
)

questions_1_5.export(
    questions_1_5_path,
    format="mp3",
    bitrate=FINAL_BITRATE,
)

questions_6_10.export(
    questions_6_10_path,
    format="mp3",
    bitrate=FINAL_BITRATE,
)

complete_test.export(
    complete_test_path,
    format="mp3",
    bitrate=FINAL_BITRATE,
)

print()
print("Finished.")
print(f"Created: {questions_1_5_path}")
print(f"Created: {questions_6_10_path}")
print(f"Created: {complete_test_path}")