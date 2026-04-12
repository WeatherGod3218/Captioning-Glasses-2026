import asyncio
import numpy as np
import whisper
import torch
import collections
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import tensorflow_hub as hub
import argparse
from fastapi import FastAPI, WebSocket
import uvicorn
from concurrent.futures import ThreadPoolExecutor
#from diart import OnlineDiarization
#from diart.sources import AudioSource

parser = argparse.ArgumentParser(description="Real-time WebSocket transcription hub.")
parser.add_argument("--model", default="tiny", choices=["tiny", "base", "small", "medium", "large", "turbo"], help="Whisper model to use.")
parser.add_argument("--non_english", action="store_true", help="Don't force the English model if it's smaller than 'large'.")
parser.add_argument("--phrase_timeout", default=1.5, type=float, help="Silence gap (sec) to trigger transcription.")
parser.add_argument("--max_duration", default=5.0, type=float, help="Maximum duration (sec) of continuous speech before transcription.")
parser.add_argument("--vad_threshold", default=0.5, type=float, help="VAD sensitivity (0.1 to 1.0) for speech detection.")
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

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected to WebSocket.")

    voiced_buffer = []
    is_speaking = False
    silence_counter = 0
    is_transcribing = False #Prevents queue bloat when still transcribing previous audio

    chunks_per_sec = SAMPLE_RATE / CHUNK_SIZE
    silence_limit = int(args.phrase_timeout * chunks_per_sec) 
    
    pre_roll = collections.deque(maxlen=15)
    yamnet_buffer = collections.deque(maxlen=15600)
    chunk_counter = 0

    loop = asyncio.get_running_loop()

    async def process_audio_task(audio_data, is_final=True):
        nonlocal is_transcribing
        is_transcribing = True
        try:
            result = await loop.run_in_executor(whisper_executor, get_speech, audio_data)
            text = result['text'].strip()
            if text:
                msg_type = "final" if is_final else "partial"
                await websocket.send_json({"type": msg_type, "text": text})
        except Exception as e:
            print(f"Transcription Error: {e}")
        finally:
            is_transcribing = False

    #Run transcription in a separate thread so it doesnt block main loop
    async def process_audio_task(audio_data):
        result = await loop.run_in_executor(whisper_executor, get_speech, audio_data)
        text = result['text'].strip()
        if text:
            await websocket.send_json({"type": "speech", "text": text})

    try:
        while True:
            raw_bytes = await websocket.receive_bytes()
            audio_chunk = np.frombuffer(raw_bytes, np.float32)
            yamnet_buffer.extend(audio_chunk)
            chunk_counter += 1
            
            if chunk_counter % 16 == 0 and len(yamnet_buffer) == 15600:
                sound_input = np.array(yamnet_buffer)
                sound, score = await loop.run_in_executor(sound_executor, get_sounds, sound_input)
                if score > 0.45 and sound not in ["Silence", "Speech"]:
                    await websocket.send_json({"type": "sound", "sound": sound})

            tensor_chunk = torch.from_numpy(audio_chunk.copy())
            speech_prob = vad_model(tensor_chunk, SAMPLE_RATE).item()

            if speech_prob > args.vad_threshold:
                if not is_speaking:
                    is_speaking = True
                    voiced_buffer.extend(list(pre_roll))
                voiced_buffer.append(audio_chunk)
                silence_counter = 0

                #Sends partial transcription every ~0.5 secs while speaking for realtime feedback without waiting for silence
                if len(voiced_buffer) % 16 == 0 and not is_transcribing:
                    full_audio = np.concatenate(voiced_buffer)
                    asyncio.create_task(process_audio_task(full_audio, is_final=False))
            else:
                if is_speaking:
                    voiced_buffer.append(audio_chunk)
                    silence_counter += 1
                    
                    if silence_counter > silence_limit:
                        is_speaking = False
                        silence_counter = 0
                        if len(voiced_buffer) > 20: 
                            full_audio = np.concatenate(voiced_buffer)
                            asyncio.create_task(process_audio_task(full_audio, is_final=True))
                        voiced_buffer = []
                else:
                    pre_roll.append(audio_chunk)

    except Exception as e:
        print(f"Client disconnected or error: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)