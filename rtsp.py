import asyncio
import numpy as np
import whisper
import torch
import ffmpeg
import collections
import tensorflow as tf
import tensorflow_hub as hub
import argparse
from fastapi import FastAPI, WebSocket
import uvicorn
from concurrent.futures import ThreadPoolExecutor

parser = argparse.ArgumentParser(description="Real-time RTSP transcription hub.")
parser.add_argument("--model", default="tiny", choices=["tiny", "base", "small", "medium", "large", "turbo"], help="Whisper model to use.")
parser.add_argument("--non_english", action="store_true", help="Don't force the English model if it's smaller than 'large'.")
parser.add_argument("--phrase_timeout", default=1.5, type=float, help="Silence gap (sec) to trigger transcription.")
parser.add_argument("--vad_threshold", default=0.5, type=float, help="VAD sensitivity (0.1 to 1.0) for speech detection.")
parser.add_argument("--rtsp_url", default="rtsp://localhost:8554/live", help="RTSP stream URL.")
args = parser.parse_args()

app = FastAPI()

#thread pools to process audio without blocking the main loop
whisper_executor = ThreadPoolExecutor(max_workers=1)
sound_executor = ThreadPoolExecutor(max_workers=1)

SAMPLE_RATE = 16000 #Whisper and YAMNet both use 16kHz audio
CHUNK_SIZE = 512 #number of samples per chunk for VAD

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device.upper()} for transcription.")

model_str = args.model
if args.model not in ["large", "turbo"] and not args.non_english:
    model_str += ".en"

print(f"Loading Whisper model: {model_str}...")
speech_model = whisper.load_model(model_str).to(device)

#SileroVAD for detecting speech so audio is processed only when speech is detected
print("Loading Silero VAD...")
vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', trust_repo=True)
(get_speech_timestamps, _, _, _, _) = utils

#YAMNet for identifying sounds
print("Loading YAMNet...")
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
class_map_path = yamnet_model.class_map_path().numpy()
with tf.io.gfile.GFile(class_map_path) as f:
    class_names = [line.split(',')[2].strip().strip('"') for line in f.read().splitlines()[1:]]

def get_speech(audio):
    return speech_model.transcribe(audio, fp16=(device=="cuda"))

def get_sounds(audio):
    scores, _, _ = yamnet_model(audio)
    class_scores = tf.reduce_mean(scores, axis=0)
    top_class = tf.argmax(class_scores)
    return class_names[top_class], class_scores[top_class].numpy()

async def transcribe_rtsp(websocket: WebSocket):
    #starts ffmpeg process to read audio from RTSP stream
    process = (
        ffmpeg
        .input(args.rtsp_url, rtsp_transport='tcp')
        .output('pipe:', format='f32le', acodec='pcm_f32le', ac=1, ar=str(SAMPLE_RATE))
        .run_async(pipe_stdout=True, pipe_stderr=True)
    )

    voiced_buffer = []
    is_speaking = False
    silence_counter = 0
    silence_limit = int(args.phrase_timeout * 32) # Approx 32 chunks per second
    
    pre_roll = collections.deque(maxlen=30)
    yamnet_buffer = collections.deque(maxlen=15600)
    chunk_counter = 0

    loop = asyncio.get_event_loop()
    print(f"Listening to RTSP stream: {args.rtsp_url}. Speak now.")

    try:
        while True:
            raw_bytes = process.stdout.read(CHUNK_SIZE * 4)
            if not raw_bytes: break

            #process audio every chunk for VAD and YAMNet
            audio_chunk = np.frombuffer(raw_bytes, np.float32)
            yamnet_buffer.extend(audio_chunk)
            chunk_counter += 1
            
            #every 16 chunks (~0.5 sec), run YAMNet to check for sounds
            if chunk_counter % 16 == 0 and len(yamnet_buffer) == 15600:
                sound_input = np.array(yamnet_buffer)
                sound, score = await loop.run_in_executor(sound_executor, get_sounds, sound_input)
                if score > 0.45 and sound not in ["Silence", "Speech"]:
                    await websocket.send_json({"type": "sound", "sound": sound})

            #runs VAD on every chunk to detect speech and buffer it until silence is detected
            tensor_chunk = torch.from_numpy(audio_chunk.copy())
            speech_prob = vad_model(tensor_chunk, SAMPLE_RATE).item()

            # Check if speech is detected based on VAD probability
            if speech_prob > args.vad_threshold:
                if not is_speaking:
                    is_speaking = True
                    voiced_buffer.extend(list(pre_roll))
                voiced_buffer.append(audio_chunk)
                silence_counter = 0
            else:
                if is_speaking:
                    voiced_buffer.append(audio_chunk)
                    silence_counter += 1
                    #If audio is silent for long enough, speech is considered ended
                    if silence_counter > silence_limit:
                        is_speaking = False
                        silence_counter = 0
                        if len(voiced_buffer) > 20: 
                            full_audio = np.concatenate(voiced_buffer)
                            result = await loop.run_in_executor(whisper_executor, get_speech, full_audio)
                            text = result['text'].strip()
                            if text:
                                await websocket.send_json({"type": "speech", "text": text})
                        voiced_buffer = []
                else:
                    pre_roll.append(audio_chunk)

    except Exception as e:
        print(f"System Error: {e}")
    finally:
        process.terminate()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    await transcribe_rtsp(websocket)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)