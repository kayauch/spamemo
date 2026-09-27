from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


@dataclass
class TranscriptionSegment:
    text: str
    start_seconds: float
    end_seconds: float


@dataclass
class TranscriptionResult:
    text: str
    language: str | None
    metrics: dict[str, float | str]
    segments: list[TranscriptionSegment] = field(default_factory=list)


class SpeechTranscriber:
    def __init__(
        self,
        model_id: str = "openai/whisper-large-v3-turbo",
        device: str = "cuda",
        min_rms: float = 0.0025,
    ):
        self.model_id = model_id
        self.device = torch.device(device)
        self.min_rms = min_rms
        self.processor = None
        self.model = None
        self.pipe = None
        self.load_metrics: dict[str, float | str] = {}

    @staticmethod
    def _audio_stats(audio: bytes) -> tuple[float, float]:
        try:
            proc = subprocess.run(
                [
                    "ffmpeg", "-loglevel", "error", "-i", "pipe:0",
                    "-f", "s16le", "-ac", "1", "-ar", "16000", "pipe:1",
                ],
                input=audio,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                timeout=30,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return 0.0, 0.0
        pcm = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32)
        if pcm.size == 0:
            return 0.0, 0.0
        pcm /= 32768.0
        rms = float(np.sqrt(np.mean(pcm * pcm)))
        return rms, float(pcm.size / 16000.0)

    @classmethod
    def audio_rms(cls, audio: bytes) -> float:
        return cls._audio_stats(audio)[0]

    def load(self) -> None:
        if self.pipe is not None:
            return
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        started = time.perf_counter()

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id,
            dtype=dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        ).to(self.device).eval()

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            dtype=dtype,
            device=0 if self.device.type == "cuda" else -1,
        )

        if self.device.type == "cuda":
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() / 2**30
        else:
            peak = 0.0
        self.load_metrics = {
            "model": self.model_id,
            "load_seconds": time.perf_counter() - started,
            "load_peak_allocated_gib": peak,
        }

    @staticmethod
    def _segments_from_output(
        output: dict,
        duration_seconds: float,
    ) -> list[TranscriptionSegment]:
        segments: list[TranscriptionSegment] = []
        for chunk in output.get("chunks") or []:
            text = str(chunk.get("text", "")).strip()
            if not text:
                continue
            timestamp = chunk.get("timestamp") or chunk.get("timestamps")
            if not isinstance(timestamp, (tuple, list)) or len(timestamp) != 2:
                continue
            start, end = timestamp
            start_value = float(start or 0.0)
            end_value = float(end if end is not None else duration_seconds)
            end_value = max(end_value, start_value)
            segments.append(
                TranscriptionSegment(
                    text=text,
                    start_seconds=max(0.0, start_value),
                    end_seconds=max(0.0, end_value),
                )
            )

        if not segments:
            text = str(output.get("text", "")).strip()
            if text:
                segments.append(
                    TranscriptionSegment(
                        text=text,
                        start_seconds=0.0,
                        end_seconds=max(duration_seconds, 0.0),
                    )
                )
        return segments

    def _transcribe(
        self,
        audio: bytes,
        language: str | None,
        return_timestamps: bool,
    ) -> TranscriptionResult:
        rms, duration_seconds = self._audio_stats(audio)
        if rms < self.min_rms:
            return TranscriptionResult(
                text="",
                language=language,
                metrics={
                    "skipped": "silence" if duration_seconds > 0 else "no_audio",
                    "rms": rms,
                    "duration_seconds": duration_seconds,
                },
            )

        self.load()
        assert self.pipe is not None
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()

        generate_kwargs = {"task": "transcribe"}
        if language:
            generate_kwargs["language"] = language
        output = self.pipe(
            audio,
            generate_kwargs=generate_kwargs,
            return_timestamps=return_timestamps,
        )

        if self.device.type == "cuda":
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() / 2**30
        else:
            peak = 0.0

        metrics = dict(self.load_metrics)
        metrics.update({
            "inference_seconds": time.perf_counter() - started,
            "inference_peak_allocated_gib": peak,
            "rms": rms,
            "duration_seconds": duration_seconds,
        })
        text = str(output.get("text", "")).strip()
        segments = (
            self._segments_from_output(output, duration_seconds)
            if return_timestamps
            else []
        )
        return TranscriptionResult(
            text=text,
            language=language,
            metrics=metrics,
            segments=segments,
        )

    def transcribe(
        self,
        audio: bytes,
        language: str | None = "ja",
    ) -> TranscriptionResult:
        return self._transcribe(audio, language, return_timestamps=False)

    def transcribe_segments(
        self,
        audio: bytes,
        language: str | None = "ja",
    ) -> TranscriptionResult:
        return self._transcribe(audio, language, return_timestamps=True)
