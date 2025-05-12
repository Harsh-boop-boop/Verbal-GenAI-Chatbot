import os
import threading
import time
import requests
import tempfile
import pygame
from queue import Queue, Empty

from dotenv import load_dotenv
from deepgram import (
    DeepgramClient,
    LiveOptions,
    LiveTranscriptionEvents,
    Microphone,
)
from together import Together

import sys

# --- Eye Display logic, NOT as a thread ---
class EyeDisplay:
    def __init__(self, screen):
        self.WIDTH, self.HEIGHT = 400, 400
        self.screen = screen
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.EYE_DISTANCE = 80
        self.EYE_Y = self.HEIGHT // 2
        self.NEUTRAL = {
            "left": [(-30, -20), (-10, -30), (10, -30), (30, -20), (30, 20), (10, 30), (-10, 30), (-30, 20)],
            "right": [(-30, -20), (-10, -30), (10, -30), (30, -20), (30, 20), (10, 30), (-10, 30), (-30, 20)],
        }
        self.EMOTIONS = {
            'Sad': {  # Sadness
                "left":  [(-18,-20), (18,-20), (10,20), (-10,20)],
                "right": [(-18,-20), (18,-20), (10,20), (-10,20)],
            },
            'Angery': {  # Anger
                "left":  [(-20,-15), (20,0), (15,15), (-15,10)],
                "right": [(20,-15), (-20,0), (-15,15), (15,10)],
            },
            'Happy': {  # Happiness
                "left":  [(-17,-10), (17,-10), (17,10), (-17,10)],
                "right": [(-17,-10), (17,-10), (17,10), (-17,10)],
            },
            'Pleading': {  # Pleading
                "left":  [(-15,-10), (15,-10), (10,10), (-10,10)],
                "right": [(-15,-10), (15,-10), (10,10), (-10,10)],
            },
            'Vulnerable': {  # Vulnerable
                "left":  [(-17,-7), (17,-7), (10,10), (-10,10)],
                "right": [(-17,-7), (17,-7), (10,10), (-10,10)],
            },
            'Despair': {  # Despair
                "left":  [(-18,0), (18,0), (10,15), (-10,15)],
                "right": [(-18,0), (18,0), (10,15), (-10,15)],
            },
            'Surprise': {  # Surprise
                "left":  [(-15,-25), (15,-25), (15,25), (-15,25)],
                "right": [(-15,-25), (15,-25), (15,25), (-15,25)],
            },
            'Disgust': {  # Disgust
                "left":  [(-18,-10), (18,0), (10,15), (-10,10)],
                "right": [(18,-10), (-18,0), (-10,15), (10,10)],
            },
            'Fear': {  # Fear
                "left":  [(-16,-18), (16,-18), (16,18), (-16,18)],
                "right": [(-16,-18), (16,-18), (16,18), (-16,18)],
            },
            'Guilty': {  # Guilty
                "left":  [(-15,10), (15,10), (15,15), (-15,15)],
                "right": [(-15,10), (15,10), (15,15), (-15,15)],
            },
            'Disappointed': {  # Disappointed
                "left":  [(-15,0), (15,0), (12,10), (-12,10)],
                "right": [(-15,0), (15,0), (12,10), (-12,10)],
            },
            'Embarrassed': {  # Embarrassed
                "left":  [(-15,2), (15,2), (12,15), (-12,15)],
                "right": [(-15,2), (15,2), (12,15), (-12,15)],
            },
            'Horrified': {  # Horrified
                "left":  [(-14,-20), (14,-20), (14,20), (-14,20)],
                "right": [(-14,-20), (14,-20), (14,20), (-14,20)],
            },
            'Skeptical': {  # Skeptical
                "left":  [(-15,-10), (15,-5), (10,10), (-10,10)],
                "right": [(-15,-5), (15,-10), (10,10), (-10,10)],
            },
            'Annoyed': {  # Annoyed
                "left":  [(-15,-7), (15,-2), (10,10), (-10,10)],
                "right": [(-15,-2), (15,-7), (10,10), (-10,10)],
            },
            'Confused': {  # Confused
                "left":  [(-13,-15), (13,-10), (13,15), (-13,10)],
                "right": [(-13,-10), (13,-15), (13,10), (-13,15)],
            },
            'Amazed': {  # Amazed
                "left":  [(-13,-22), (13,-22), (13,22), (-13,22)],
                "right": [(-13,-22), (13,-22), (13,22), (-13,22)],
            },
            'Furious': {  # Furious
                "left":  [(-17,-12), (17,4), (13,13), (-13,7)],
                "right": [(17,-12), (-17,4), (-13,13), (13,7)],
            },
            'Suspicious': {  # Suspicious
                "left":  [(-17,-5), (17,-10), (13,7), (-13,10)],
                "right": [(-17,-10), (17,-5), (13,10), (-13,7)],
            },
            'Rejected': {  # Rejected
                "left":  [(-15,8), (15,8), (12,15), (-12,15)],
                "right": [(-15,8), (15,8), (12,15), (-12,15)],
            },

            'Tired': {  # Tired
                "left":  [(-14,8), (14,5), (14,10), (-14,13)],
                "right": [(-14,5), (14,8), (14,13), (-14,10)],
            },
            'Asleep': {  # Asleep
                "left":  [(-15,15), (15,15), (15,17), (-15,17)],
                "right": [(-15,15), (15,15), (15,17), (-15,17)],
            },
        }
        self.current_emotion = 'neutral'
        self.target_emotion = 'neutral'
        self.transition_progress = 0.5
        self.transition_time = 0.4
        self.last_time = time.time()
        # For animation state
        self.left_now = self.NEUTRAL['left']
        self.right_now = self.NEUTRAL['right']

    def make_eye(self, center, points):
        return [(center[0] + x, center[1] + y) for (x, y) in points]

    def lerp_points(self, points1, points2, t):
        n = max(len(points1), len(points2))
        p1 = list(points1) + [points1[-1]] * (n - len(points1))
        p2 = list(points2) + [points2[-1]] * (n - len(points2))
        return [
            (
                p1[i][0] * (1 - t) + p2[i][0] * t,
                p1[i][1] * (1 - t) + p2[i][1] * t
            )
            for i in range(n)
        ]

    def draw_eyes(self):
        self.screen.fill(self.BLACK)
        left_eye_center = (self.WIDTH // 2 - self.EYE_DISTANCE // 2, self.EYE_Y)
        right_eye_center = (self.WIDTH // 2 + self.EYE_DISTANCE // 2, self.EYE_Y)
        pygame.draw.polygon(self.screen, self.WHITE, self.make_eye(left_eye_center, self.left_now))
        pygame.draw.polygon(self.screen, self.WHITE, self.make_eye(right_eye_center, self.right_now))
        pygame.display.flip()

    def update_emotion(self, new_emotion):
        # Start animation to new emotion
        if new_emotion not in self.EMOTIONS:
            new_emotion = 'Happy'
        self.target_emotion = new_emotion
        self.transition_progress = 0.0
        self.last_time = time.time()

    def animate(self):
        # Animate transition if in progress
        if self.transition_progress < 1.0:
            now = time.time()
            dt = now - self.last_time
            self.last_time = now
            self.transition_progress += dt / self.transition_time
            self.transition_progress = min(self.transition_progress, 1.0)
            left_from = self.NEUTRAL['left'] if self.current_emotion == 'neutral' else self.EMOTIONS.get(self.current_emotion, self.NEUTRAL)['left']
            right_from = self.NEUTRAL['right'] if self.current_emotion == 'neutral' else self.EMOTIONS.get(self.current_emotion, self.NEUTRAL)['right']
            left_to = self.EMOTIONS.get(self.target_emotion, self.NEUTRAL)['left']
            right_to = self.EMOTIONS.get(self.target_emotion, self.NEUTRAL)['right']
            self.left_now = self.lerp_points(left_from, left_to, self.transition_progress)
            self.right_now = self.lerp_points(right_from, right_to, self.transition_progress)
            if self.transition_progress == 1.0:
                self.current_emotion = self.target_emotion
        else:
            self.left_now = self.EMOTIONS.get(self.current_emotion, self.NEUTRAL)['left']
            self.right_now = self.EMOTIONS.get(self.current_emotion, self.NEUTRAL)['right']


# --- AI Bot and Audio ---
api_key = "ADD API KEY HERE"
client = Together(api_key=api_key)

def get_ai_response(user_question):
    stream = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        messages=[
            {"role": "system", "content": "You are a representative of “The Hong Kong University of Science and Technology” - school of engineering named birdie. Your purpose is to try and detect how the people talking to you are feeling and then respond to them accordingly. You try to be as helpful and polite as possible, unless you get angry. You are also a nerd and love “Hong Kong University of Science and Technology” and engineering. You must keep your answers short."},
            {"role": "user", "content": user_question}],
        stream=True,
    )
    response = ""
    for chunk in stream:
        response += chunk.choices[0].delta.content or ""
    return response

def get_ai_response_emotion(user_question):
    stream = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        messages=[
            {"role": "system", "content": "You are a emotion robot, you will recieve a user input, and then react accordingly. Your output will be only 1 word and it will be from these options. Happy, Sad, Angry, Pleading, Vulnerable, Despair, Surprise, Disgust, Fear, Guilty, Disappointed, Embarrassed, Horrified, Skeptical, Annoyed, Confused, Amazed, Furious, Suspicious, Rejected, Tired or Asleep"},
            {"role": "user", "content": user_question}],
        stream=True,
    )
    response = ""
    for chunk in stream:
        response += chunk.choices[0].delta.content or ""
    # Clean up to just the word
    return response.strip().split()[0].capitalize()

load_dotenv()
DEEPGRAM_API_KEY = "ADD API KEY HERE"
DEEPGRAM_TTS_URL = "https://api.deepgram.com/v1/speak?model=aura-helios-en"
HEADERS = {
    "Authorization": f"Token {DEEPGRAM_API_KEY}",
    "Content-Type": "application/json"
}

# Global flag to control microphone state
mute_microphone = threading.Event()

def synthesize_audio(text):
    payload = {"text": text}
    resp = requests.post(DEEPGRAM_TTS_URL, headers=HEADERS, json=payload)
    resp.raise_for_status()
    return resp.content

def play_audio(audio_bytes):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    temp_file.write(audio_bytes)
    temp_file.close()
    pygame.mixer.init()
    pygame.mixer.music.load(temp_file.name)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    pygame.mixer.music.stop()
    pygame.mixer.quit()
    os.remove(temp_file.name)
    mute_microphone.clear()

# --- AI/Audio Worker Thread ---
def ai_audio_worker(emotion_queue):
    try:
        dg_client = DeepgramClient(DEEPGRAM_API_KEY)
        connection = dg_client.listen.websocket.v("1")  # Modern, supported API
        is_finals = []

        def on_open(self, open, **kwargs):
            print("Connection Open")

        def on_message(self, result, **kwargs):
            nonlocal is_finals
            global microphone
            if mute_microphone.is_set():
                return  # Ignore messages while microphone is muted

            transcript = result.channel.alternatives[0].transcript
            if len(transcript) == 0:
                return
            if result.is_final:
                is_finals.append(transcript)
            if result.speech_final:
                utterance = " ".join(is_finals)
                print(f"User said: {utterance}")
                is_finals = []
                response_text = get_ai_response(utterance)
                emotion_text = get_ai_response_emotion(utterance)
                print(f"Detected emotion: {emotion_text}")
                # Send the emotion to the main thread
                emotion_queue.put(emotion_text)
                print(emotion_text)
                audio_data = synthesize_audio(response_text)
                mute_microphone.set()
                microphone.mute()
                play_audio(audio_data)
                time.sleep(0.3)
                microphone.unmute()
            else:
                print(f"Interim: {transcript}")

        def on_close(self, close, **kwargs):
            print("Connection Closed")

        def on_error(self, error, **kwargs):
            print(f"Error: {error}")

        connection.on(LiveTranscriptionEvents.Open, on_open)
        connection.on(LiveTranscriptionEvents.Transcript, on_message)
        connection.on(LiveTranscriptionEvents.Close, on_close)
        connection.on(LiveTranscriptionEvents.Error, on_error)

        options = LiveOptions(
            model="nova-2",
            language="en-US",
            smart_format=True,
            encoding="linear16",
            channels=1,
            sample_rate=16000,
            interim_results=True,
            utterance_end_ms="1000",
            vad_events=True,
            endpointing=500,
        )

        addons = {"no_delay": "true"}

        print("\nPress Enter to stop recording...\n")
        if not connection.start(options, addons=addons):
            print("Failed to connect to Deepgram. Please check your API key.")
            return

        global microphone
        microphone = Microphone(connection.send)  # Must be local to handler
        microphone.start()

        input("")  # Wait for user to press Enter
        microphone.finish()
        connection.finish()
        print("Finished")

    except Exception as e:
        print(f"Fatal error: {e}")

# --- Main Integration ---
def main():
    # Initialize Pygame and create the window
    pygame.init()
    WIDTH, HEIGHT = 400, 400
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Expressive Eyes")
    clock = pygame.time.Clock()
    eyes = EyeDisplay(screen)

    # Show empty screen and wait for Enter key press
    waiting_for_enter = True
    while waiting_for_enter:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:  # Enter key pressed
                    waiting_for_enter = False

        screen.fill((0, 0, 0))  # Fill screen black (empty)
        pygame.display.flip()
        clock.tick(30)

    # Once Enter is pressed, start the AI/audio worker thread and main loop
    emotion_queue = Queue()
    worker = threading.Thread(target=ai_audio_worker, args=(emotion_queue,), daemon=True)
    worker.start()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Check for emotion updates from the queue
        try:
            while True:
                new_emotion = emotion_queue.get_nowait()
                eyes.update_emotion(new_emotion)
        except Empty:
            pass

        eyes.animate()
        eyes.draw_eyes()
        clock.tick(60)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
