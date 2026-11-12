"""On-robot speech-to-text. Each robot has a microphone and runs ASR itself, then sends only the
text to the console, so raw audio never leaves the robot."""
from __future__ import annotations

import os


class TextASR:
    """For the simulation: the dashboard's typed sentence stands in for what the microphone heard."""

    name = "text"

    def transcribe(self, audio):
        return str(audio).strip()


class WhisperASR:
    """faster-whisper running on the robot (e.g. `small`, int8 on CPU)."""

    def __init__(self, model="small"):
        from faster_whisper import WhisperModel  # optional dependency
        self.model = WhisperModel(model, device="cpu", compute_type="int8")
        self.name = f"whisper:{model}"

    def transcribe(self, audio):
        if not os.path.exists(str(audio)):
            return str(audio).strip()  # typed text from the dashboard
        segments, _ = self.model.transcribe(str(audio), language="zh")
        return "".join(s.text for s in segments).strip()


def make_asr(cfg):
    if cfg.get("backend") == "whisper":
        try:
            return WhisperASR(cfg.get("model", "small"))
        except Exception as e:
            print(f"[asr] faster-whisper unavailable ({e}); using TextASR")
    return TextASR()
