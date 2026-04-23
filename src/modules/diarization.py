import time
import numpy as np
import collections
from logging import Logger, getLogger
from pyannote.core import Annotation, Segment
from diart import SpeakerDiarization, SpeakerDiarizationConfig
from diart.sources import AudioSource
from diart.inference import StreamingInference

from config import HF_TOKEN, SAMPLE_RATE


# Audio source to feed websocket audio into Diart
class WebSocketAudioSource(AudioSource):
    def __init__(self, sample_rate):
        super().__init__(uri="websocket_stream", sample_rate=sample_rate)

    def read(self):
        pass

    def close(self):
        self.stream.on_completed()

    def push_audio(self, chunk: np.ndarray):
        self.stream.on_next(chunk.reshape(1, -1))


logger: Logger = getLogger(__name__)
logger.info("Loading Diart (Pyannote)...")

diart_config: SpeakerDiarizationConfig = SpeakerDiarizationConfig(
    duration=2.0, step=0.3, latency="min", sample_rate=SAMPLE_RATE, hf_token=HF_TOKEN
)
diarization: SpeakerDiarization = SpeakerDiarization(diart_config)

audio_source: WebSocketAudioSource = WebSocketAudioSource(SAMPLE_RATE)
pipeline: StreamingInference = StreamingInference(diarization, audio_source)


speaker_timeline: collections.deque[tuple[float, str]] = collections.deque(
    maxlen=50
)  # store recent speaker labels with timestamps


def on_diarization_update(result: tuple[Annotation] | Annotation) -> None:
    """
    Adds result processed from the Audio Pipeline to be added to the speaker queue

    Arguments:
        result (tuple[Annotation] | Annotation): Either a tuple or just an annotation, depending on current buffer
    """

    annotation: Annotation = result[0] if isinstance(result, tuple) else result

    if not hasattr(annotation, "labels") or not annotation.labels():
        return

    try:
        tracks: list[tuple[Segment, str, str]] = list(
            annotation.itertracks(yield_label=True)
        )
        if not tracks:
            return
        # Get most recent speaker segment
        latest_track: tuple[Segment, str, str] = max(tracks, key=lambda x: x[0].end)
        speaker_timeline.append((time.monotonic(), latest_track[2]))
    except Exception:
        return


def get_speaker_at(timestamp: float, max_age: float = 1.5) -> str:
    """
    Finds the most recent speaker at or before the given timestamp, or 00 if none is found

    Arguments:
        timestamp (float): The timestamp of the speaker to look for
        max_age (float): The max age to look back, defaults to 1.5

    Returns:
        str: The label of the speaker, or SPEAKER_00 if none is found
    """

    best: str | None = None

    for ts, spk in reversed(speaker_timeline):
        if ts <= timestamp + max_age:
            best = spk
            break
    return best or "SPEAKER_00"
