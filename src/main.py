import asyncio
import numpy as np
import collections
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")  # Forces TF to use CPU
import tensorflow_hub as hub
import argparse
import os
import huggingface_hub
import time

# Torch patch to make sure model loading works on older versions
import torch

from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
import uvicorn
from concurrent.futures import ThreadPoolExecutor
from faster_whisper import WhisperModel
from diart import SpeakerDiarization, SpeakerDiarizationConfig
from diart.sources import AudioSource
from diart.inference import StreamingInference

from logging import getLogger, Logger
from config import BASE_DIR, HF_TOKEN

logger: Logger = getLogger(__name__)
# huggingface patch to support old token arg
_old_download = huggingface_hub.hf_hub_download


def _patched_download(*args, **kwargs):
    if "use_auth_token" in kwargs:
        kwargs["token"] = kwargs.pop("use_auth_token")
    return _old_download(*args, **kwargs)


huggingface_hub.hf_hub_download = _patched_download

_old_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _old_torch_load(*args, **kwargs)


torch.load = _patched_torch_load

parser = argparse.ArgumentParser(description="Real-time WebSocket transcription hub.")
parser.add_argument(
    "--phrase_timeout",
    default=0.6,
    type=float,
    help="Silence gap (sec) to trigger final transcription.",
)
parser.add_argument(
    "--max_duration",
    default=3.0,
    type=float,
    help="Max duration before forcing a final result.",
)
parser.add_argument(
    "--vad_threshold",
    default=0.4,
    type=float,
    help="VAD sensitivity (lower = more sensitive).",
)
args = parser.parse_known_args()

app = FastAPI()

if os.path.exists(os.path.join(BASE_DIR, "docs")):
	logger.info("Documentation directory found, setting up documentation endpoint!")

	app.mount(
		"/docs", StaticFiles(directory=os.path.join(BASE_DIR, "docs")), name="docs"
	)

	@app.get("/docs", include_in_schema=False)
	async def docs_redirect():
		# Mkdocs links dynamically and not being on the direct index.html causes issues
		return RedirectResponse(url="/docs/index.html")

else:
	logger.warning("Documentation directory not found, skipping documentation setup!")


gpu_lock = asyncio.Lock()

whisper_executor = ThreadPoolExecutor(max_workers=1)
sound_executor = ThreadPoolExecutor(max_workers=1)
diart_executor = ThreadPoolExecutor(max_workers=1)

SAMPLE_RATE = 16000
CHUNK_SIZE = 2048

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device.upper()} for transcription.")

print("Loading Whisper...")
compute_type = "float16" if device == "cuda" else "int8"
speech_model = WhisperModel(
    "deepdml/faster-whisper-large-v3-turbo-ct2", device=device, compute_type=compute_type
)

print("Loading Diart (Pyannote)...")
diart_config = SpeakerDiarizationConfig(
    duration=2.0, step=0.3, latency="min", sample_rate=SAMPLE_RATE, hf_token=HF_TOKEN
)
diarization = SpeakerDiarization(diart_config)


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


audio_source = WebSocketAudioSource(SAMPLE_RATE)
pipeline = StreamingInference(diarization, audio_source)


speaker_timeline = collections.deque(
    maxlen=50
)  # store recent speaker labels with timestamps


def on_diarization_update(result):
    annotation = result[0] if isinstance(result, tuple) else result
    if not hasattr(annotation, "labels") or not annotation.labels():
        return
    try:
        tracks = list(annotation.itertracks(yield_label=True))
        if not tracks:
            return
        # Get most recent speaker segment
        latest_track = max(tracks, key=lambda x: x[0].end)
        speaker_timeline.append((time.monotonic(), latest_track[2]))
    except Exception:
        pass


def get_speaker_at(timestamp, max_age=1.5):
    """Finds the most recent speaker at or before the given timestamp."""
    best = None
    for ts, spk in reversed(speaker_timeline):
        if ts <= timestamp + max_age:
            best = spk
            break
    return best or "SPEAKER_00"


pipeline.stream.subscribe(on_diarization_update)

print("Loading Silero VAD...")
vad_model, utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True
)
vad_model = vad_model.to(device)

print("Loading YAMNet...")
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
class_map_path = yamnet_model.class_map_path().numpy()
with tf.io.gfile.GFile(class_map_path) as f:
    class_names = [
        line.split(",")[2].strip().strip('"') for line in f.read().splitlines()[1:]
    ]


def get_speech(audio, is_final=True):
    beam = 5 if is_final else 1  # more accurate for final, faster for partial
    segments, _ = speech_model.transcribe(
        audio,
        beam_size=beam,
        language="en",
        condition_on_previous_text=True,
        temperature=[0.0, 0.2, 0.4],
        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=500, speech_pad_ms=200, threshold=0.2
        ),
        no_speech_threshold=0.65,
        log_prob_threshold=-2.0,
        compression_ratio_threshold=2.0,
        repetition_penalty=1.2,
    )
    text = " ".join([segment.text for segment in segments])
    return {"text": text}


def get_sounds(audio):
    scores, _, _ = yamnet_model(audio)
    class_scores = tf.reduce_mean(scores, axis=0)
    top_class = tf.argmax(class_scores)
    return class_names[top_class], class_scores[top_class].numpy()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected.")

    voiced_buffer = []  # audio chunks for current speech
    is_speaking = False
    silence_counter = 0
    is_transcribing = False
    utterance_start_time = time.monotonic()

    chunks_per_sec = SAMPLE_RATE / CHUNK_SIZE
    silence_limit = int(args.phrase_timeout * chunks_per_sec)

    pre_roll = collections.deque(
        maxlen=10
    )  # keeps audio just before speech starts to avoid cutting off beginning of phrases
    yamnet_buffer = collections.deque(maxlen=16384)
    chunk_counter = 0

    loop = asyncio.get_running_loop()

    async def process_audio_task(audio_data, speaker_at_capture, is_final=True):
        nonlocal is_transcribing

        if audio_data is None or len(audio_data) == 0:
            return

        if gpu_lock.locked() and not is_final:
            return

        async with gpu_lock:
            is_transcribing = True
            try:
                result = await loop.run_in_executor(
                    whisper_executor, get_speech, audio_data, is_final
                )
                text = result["text"].strip()
                if text:
                    msg_type = "final" if is_final else "partial"
                    await websocket.send_json(
                        {"type": msg_type, "text": text, "speaker": speaker_at_capture}
                    )
            except Exception as e:
                print(f"Transcription Error: {e}")
            finally:
                is_transcribing = False

    try:
        while True:
            raw_bytes = await websocket.receive_bytes()
            audio_chunk = np.frombuffer(raw_bytes, np.float32).copy()

            loop.run_in_executor(
                diart_executor, audio_source.push_audio, audio_chunk.copy()
            )

            yamnet_buffer.extend(audio_chunk)
            chunk_counter += 1

            if chunk_counter % 8 == 0 and len(yamnet_buffer) == 16384:

                async def ps(buf):
                    try:
                        s, sc = await loop.run_in_executor(
                            sound_executor, get_sounds, buf
                        )
                        if sc > 0.45 and s not in ["Silence", "Speech"]:
                            await websocket.send_json({"type": "sound", "sound": s})
                    except:
                        pass

                asyncio.create_task(ps(np.array(yamnet_buffer)))

            def check_vad():
                with torch.no_grad():
                    sub_chunks = torch.from_numpy(audio_chunk).to(device).split(512)
                    max_prob = 0.0
                    for sub in sub_chunks:
                        prob = vad_model(sub, SAMPLE_RATE).item()
                        max_prob = max(max_prob, prob)
                    return max_prob

            speech_prob = await loop.run_in_executor(None, check_vad)

            if speech_prob > args.vad_threshold:
                if not is_speaking:
                    is_speaking = True
                    utterance_start_time = time.monotonic()
                    voiced_buffer.extend(list(pre_roll))
                voiced_buffer.append(audio_chunk)
                silence_counter = 0

                # Sends partial transcription every 4 chunks while speaking
                if len(voiced_buffer) % 4 == 0:
                    speaker_snapshot = get_speaker_at(utterance_start_time)
                    asyncio.create_task(
                        process_audio_task(
                            np.concatenate(voiced_buffer),
                            speaker_snapshot,
                            is_final=False,
                        )
                    )

                # Force final transcriptions if buffer gets too long
                if (len(voiced_buffer) * CHUNK_SIZE) / SAMPLE_RATE >= args.max_duration:
                    speaker_snapshot = get_speaker_at(utterance_start_time)
                    asyncio.create_task(
                        process_audio_task(
                            np.concatenate(voiced_buffer),
                            speaker_snapshot,
                            is_final=True,
                        )
                    )
                    voiced_buffer = []
                    is_speaking = False
                    pre_roll.clear()
            else:
                # Silence handling
                if is_speaking:
                    voiced_buffer.append(audio_chunk)
                    silence_counter += 1
                    if silence_counter > silence_limit:
                        is_speaking = False
                        if len(voiced_buffer) > 4:
                            speaker_snapshot = get_speaker_at(utterance_start_time)
                            asyncio.create_task(
                                process_audio_task(
                                    np.concatenate(voiced_buffer),
                                    speaker_snapshot,
                                    is_final=True,
                                )
                            )
                        voiced_buffer = []
                        pre_roll.clear()
                else:
                    pre_roll.append(audio_chunk)

    except Exception as e:
        print(f"WS Disconnected: {e}")
