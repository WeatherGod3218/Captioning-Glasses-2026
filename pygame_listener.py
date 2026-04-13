import asyncio
import websockets
import json
import pyaudio
import pygame
import sys

FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 16000
CHUNK = 512

pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Transcription Display")
font = pygame.font.SysFont("arial", 32)

state = {
    "finals": [], #List of finalized sentences
    "partial": "", #The current "in-progress" sentence
    "sound": "" #Last detected environmental sound
}

MAX_SENTENCE_HISTORY = 50 

def wrap_text(text, font, max_width):
    """Splits a string into a list of strings that fit within max_width."""
    words = text.split(' ')
    lines = []
    current_line = []
    
    for word in words:
        test_line = ' '.join(current_line + [word])
        width, _ = font.size(test_line)
        
        if width <= max_width:
            current_line.append(word)
        else:
            lines.append(' '.join(current_line))
            current_line = [word]
            
    if current_line:
        lines.append(' '.join(current_line))
        
    return lines

async def send_audio(websocket):
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            await websocket.send(data)
            await asyncio.sleep(0.001)
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

async def receive_text(websocket):
    while True:
        data = await websocket.recv()
        msg = json.loads(data)
        
        if msg['type'] == "partial":
            state["partial"] = msg['text']
            
        elif msg['type'] == "final":
            state["finals"].append(msg['text'])
            state["partial"] = ""
            
            if len(state["finals"]) > MAX_SENTENCE_HISTORY:
                state["finals"].pop(0) 
                
        elif msg['type'] == "sound":
            state["sound"] = msg['sound']

async def pygame_loop():
    max_text_width = WIDTH - 40
    line_height = 40
    usable_height = HEIGHT - 80 
    max_display_lines = usable_height // line_height
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        screen.fill((15, 15, 15)) 
        
        all_rendered_lines = []
        for text in state["finals"]:
            all_rendered_lines.extend(wrap_text(text, font, max_text_width))
            
        if state["partial"]:
            all_rendered_lines.extend(wrap_text(state["partial"] + "...", font, max_text_width))
            
        visible_lines = all_rendered_lines[-max_display_lines:]

        y_offset = 20
        for line in visible_lines:
            text_surface = font.render(line, True, (240, 240, 240))
            screen.blit(text_surface, (20, y_offset))
            y_offset += line_height

        if state["sound"]:
            sound_surface = font.render(f"[{state['sound'].upper()}]", True, (100, 150, 255))
            screen.blit(sound_surface, (20, HEIGHT - 50))

        pygame.display.flip()
        await asyncio.sleep(0.01)

async def main():
    uri = "wss://practitioner-watching-verified-assisted.trycloudflare.com/ws"
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected to WebSocket. Launching Pygame UI...")
            await asyncio.gather(
                send_audio(websocket),
                receive_text(websocket),
                pygame_loop()
            )
    except Exception as e:
        print(f"Connection Error: {e}")
        pygame.quit()

if __name__ == "__main__":
    asyncio.run(main())