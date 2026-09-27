import io
import math
import struct
import wave

from spatial_memory.speech import SpeechTranscriber


def wav_bytes(samples: list[int], rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(b"".join(struct.pack("<h", s) for s in samples))
    return buf.getvalue()


def test_silence_is_skipped_without_loading_model() -> None:
    audio = wav_bytes([0] * 16000)
    transcriber = SpeechTranscriber(min_rms=0.0025)
    result = transcriber.transcribe(audio, language="ja")
    assert result.text == ""
    assert result.metrics["skipped"] == "silence"
    assert transcriber.pipe is None


def test_tone_has_nonzero_rms() -> None:
    samples = [
        int(6000 * math.sin(2 * math.pi * 440 * i / 16000))
        for i in range(16000)
    ]
    audio = wav_bytes(samples)
    assert SpeechTranscriber.audio_rms(audio) > 0.05


def test_timestamp_chunks_are_normalized_to_segments() -> None:
    output = {
        "text": "これは父の眼鏡です。これは予備の鍵です。",
        "chunks": [
            {"text": "これは父の眼鏡です。", "timestamp": (0.5, 2.0)},
            {"text": "これは予備の鍵です。", "timestamp": (4.0, None)},
        ],
    }
    segments = SpeechTranscriber._segments_from_output(output, duration_seconds=6.0)
    assert [segment.text for segment in segments] == [
        "これは父の眼鏡です。",
        "これは予備の鍵です。",
    ]
    assert segments[0].start_seconds == 0.5
    assert segments[0].end_seconds == 2.0
    assert segments[1].start_seconds == 4.0
    assert segments[1].end_seconds == 6.0
