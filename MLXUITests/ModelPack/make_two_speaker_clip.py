#!/usr/bin/env python3
"""MP4-3a: builds the two-speaker parity clip for the Sortformer diarization tests.

Run:  python3 -I MLXUITests/ModelPack/make_two_speaker_clip.py
Needs macOS `say` and `afconvert` (both ship with macOS); nothing else, python stdlib only.

Eight alternating turns, voice A (Samantha, US female) and voice B (Daniel, UK male). Each turn is rendered with
`say`, converted to 16 kHz mono 16-bit PCM, and trimmed of the leading and trailing silence `say` adds (a fixed
amplitude threshold). The clip is then assembled from a fixed lead-in, the trimmed turns and a fixed gap between turns,
all in whole samples, so the ground-truth turn times are exact sample arithmetic and the build is repeatable on the same
macOS voices. Output, next to this script:

  two_speakers.wav          16 kHz mono 16-bit PCM
  two_speakers_truth.json   the turn list: speaker (0 = Samantha, 1 = Daniel), start and end in seconds

TEST TARGET ONLY. This clip is never part of the app bundle, Fixtures/CatFlow or Resources/Gallery.
"""
import json
import os
import struct
import subprocess
import sys
import tempfile
import wave

RATE = 16_000
LEAD_IN_S = 0.5
GAP_S = 1.0
TAIL_S = 0.5
TRIM_AMPLITUDE = 300          # of 32768; below this at the edges counts as silence

VOICES = {0: "Samantha", 1: "Daniel"}

TURNS = [
    (0, "Good morning everyone. Thanks for joining the call today, I want to start with the schedule for the release."),
    (1, "Morning. The build is nearly ready, but the last tests are still running and I expect them to finish this afternoon."),
    (0, "That sounds fine. Did the team finish the documentation for the new settings screen?"),
    (1, "Almost. Two pages are left, and I will send them to you for review before the end of the day."),
    (0, "Great. Then let us plan to ship on Thursday, as long as nothing unexpected turns up in the final checks."),
    (1, "Agreed. I will also prepare the release notes, and mention the performance improvements we measured last week."),
    (0, "Perfect. Please keep the notes short, because most people only read the first few lines."),
    (1, "Understood. I will keep them brief. Thank you all, and have a good rest of the day."),
]


def render_turn(voice, text, workdir, index):
    aiff = os.path.join(workdir, f"turn{index}.aiff")
    wav = os.path.join(workdir, f"turn{index}.wav")
    subprocess.run(["say", "-v", voice, "-o", aiff, text], check=True)
    subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@%d" % RATE, "-c", "1", aiff, wav], check=True)
    with wave.open(wav, "rb") as w:
        assert w.getframerate() == RATE and w.getnchannels() == 1 and w.getsampwidth() == 2
        raw = w.readframes(w.getnframes())
    samples = list(struct.unpack("<%dh" % (len(raw) // 2), raw))
    start = 0
    while start < len(samples) and abs(samples[start]) < TRIM_AMPLITUDE:
        start += 1
    end = len(samples)
    while end > start and abs(samples[end - 1]) < TRIM_AMPLITUDE:
        end -= 1
    return samples[start:end]


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    out = []
    truth = []
    out.extend([0] * int(LEAD_IN_S * RATE))
    with tempfile.TemporaryDirectory() as workdir:
        for i, (speaker, text) in enumerate(TURNS):
            turn = render_turn(VOICES[speaker], text, workdir, i)
            if i > 0:
                out.extend([0] * int(GAP_S * RATE))
            start = len(out)
            out.extend(turn)
            truth.append({"speaker": speaker, "start": start / RATE, "end": len(out) / RATE, "voice": VOICES[speaker]})
    out.extend([0] * int(TAIL_S * RATE))
    wav_path = os.path.join(here, "two_speakers.wav")
    with wave.open(wav_path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(struct.pack("<%dh" % len(out), *out))
    doc = {
        "clip": "two_speakers.wav",
        "sampleRate": RATE,
        "durationSeconds": len(out) / RATE,
        "note": "Turn times are the trimmed speech span of each say-rendered turn (sample arithmetic from "
                "make_two_speaker_clip.py). Pauses inside a turn are part of the turn. Voice 0 is Samantha, voice 1 is Daniel; "
                "Sortformer's own speaker numbering is arrival order, so compare by alternation, not by index.",
        "turns": truth,
    }
    with open(os.path.join(here, "two_speakers_truth.json"), "w") as f:
        json.dump(doc, f, indent=2)
        f.write("\n")
    print(json.dumps(doc, indent=2))


if __name__ == "__main__":
    sys.exit(main())
