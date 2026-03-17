import asyncio
import websockets
import json

async def listen():
    uri = "ws://127.0.0.1:8000/ws"
    async with websockets.connect(uri) as websocket:
        print("Connected to websocket. Listening...")
        while True:
            data = await websocket.recv()
            msg = json.loads(data)
            
            if msg['type'] == "speech":
                print(f"> {msg['text']}")
            elif msg['type'] == "sound":
                print(f"[{msg['sound'].upper()}]")

asyncio.run(listen())