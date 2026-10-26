#!/usr/bin/env python3
"""Reassemble an existing IELTS Listening Part 2 with pitch-preserved speaker slowdown.

Offline: no Gemini client, no API calls, no transcript files required.

Example:
    python adjust_part2_speed.py 2_cycle_club_demo --speed 0.90
    python adjust_part2_speed.py 2_book_festival --speed 0.93 --raw-dir raw

Requires: pydub and ffmpeg.
"""

import argparse
import io
import shutil
import subprocess
from pathlib import Path

from pydub import AudioSegment

NARRATOR_WAVS = ("intro", "start_1", "middle", "start_2", "end")
SPEAKER_WAVS = ("dialogue_q1_5.wav", "dialogue_q6_10.wav")


def slow_speaker(path: Path, speed: float, ffmpeg: str) -> AudioSegment:
    if speed == 1.0:
        return AudioSegment.from_wav(path)
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
           "-i", str(path), "-vn", "-af", f"atempo={speed:.6f}",
           "-f", "wav", "pipe:1"]
    result = subprocess.run(cmd, capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError(f"Could not slow {path}: {result.stderr.decode('utf-8', 'replace')}")
    audio = AudioSegment.from_file(io.BytesIO(result.stdout), format="wav")
    before = AudioSegment.from_wav(path)
    print(f"  {path.name}: {before.duration_seconds:.1f}s -> {audio.duration_seconds:.1f}s")
    return audio


def run(args):
    if not (0.5 <= args.speed <= 1.0):
        raise ValueError("--speed must be between 0.50 and 1.00; e.g. 0.90")
    base = Path(__file__).resolve().parent
    test_dir = base / "tests" / args.test
    raw_dir = test_dir / "audio" / args.raw_dir
    final_dir = test_dir / "audio" / args.final_dir
    required = list(SPEAKER_WAVS) + [f"{name}.wav" for name in NARRATOR_WAVS]
    missing = [name for name in required if not (raw_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing recordings in {raw_dir}: {', '.join(missing)}. "
            "Change --raw-dir if your WAV files are in another folder.")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required. Install it before running this script.")
    final_dir.mkdir(parents=True, exist_ok=True)

    print(f"Test: {args.test}; speaker speed: {args.speed:.2f}x")
    speaker1 = slow_speaker(raw_dir / SPEAKER_WAVS[0], args.speed, ffmpeg)
    speaker2 = slow_speaker(raw_dir / SPEAKER_WAVS[1], args.speed, ffmpeg)
    narrator = {name: AudioSegment.from_wav(raw_dir / f"{name}.wav") for name in NARRATOR_WAVS}
    silence = lambda ms: AudioSegment.silent(duration=ms, frame_rate=24_000)

    first = (narrator["intro"] + silence(args.beginning_pause)
             + narrator["start_1"] + silence(args.short_pause) + speaker1)
    second = (narrator["middle"] + silence(args.middle_pause)
              + narrator["start_2"] + silence(args.short_pause)
              + speaker2 + silence(args.short_pause) + narrator["end"])
    complete = first + silence(args.short_pause) + second

    tag = f"speed{round(args.speed * 100):03d}"
    outputs = {
        f"{args.test}_questions_11-14_{tag}.mp3": first,
        f"{args.test}_questions_15-20_{tag}.mp3": second,
        f"{args.test}_complete_{tag}.mp3": complete,
    }
    for filename, audio in outputs.items():
        destination = final_dir / filename
        audio.export(destination, format="mp3", bitrate=args.bitrate)
        print(f"  Created: {destination} ({audio.duration_seconds:.1f}s)")
    print("Finished. Original WAV files are unchanged. No API calls made.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("test", help="Folder name under tests/, e.g. 2_cycle_club_demo")
    parser.add_argument("--speed", type=float, default=0.95, help="Speaker speed (default 0.95; 0.90 = 10%% slower)")
    parser.add_argument("--raw-dir", default="raw_38", help="WAV folder name under audio/ (default: raw_38)")
    parser.add_argument("--final-dir", default="final_38", help="MP3 output folder name under audio/ (default: final_38)")
    parser.add_argument("--beginning-pause", type=int, default=33000, help="Opening preparation silence, ms (default 33000)")
    parser.add_argument("--middle-pause", type=int, default=26000, help="Second preparation silence, ms (default 26000)")
    parser.add_argument("--short-pause", type=int, default=1000, help="Narrator transition silence, ms (default 1000)")
    parser.add_argument("--bitrate", default="192k", help="MP3 bitrate (default 192k)")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
