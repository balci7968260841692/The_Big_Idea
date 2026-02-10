# voice_control/voice_keyboard.py
import json
import queue
import threading
import time
from pathlib import Path
import numpy as np
import sounddevice as sd
from vosk import Model, KaldiRecognizer
from pynput import keyboard
from pynput.mouse import Button, Controller as MouseController
# ---------- SETTINGS ----------
VOSK_MODEL_PATH = "vosk-model-small-en-us-0.15" # folder you downloaded/unzipped
PUSH_TO_TALK_KEY = keyboard.Key.f9 # change if you want
SAMPLE_RATE = 16000
CHANNELS = 1
MAX_RECORD_SECONDS = 8.0
BLOCKSIZE = 4000 # FIX #8: fixed blocksize (0.25s at 16kHz) — more stable than blocksize=0
# ----------------------------
_audio_q: "queue.Queue[np.ndarray]" = queue.Queue()
# FIX #1 (thread safety): Use a threading.Lock + a plain bool protected by it,
# instead of a bare global mutated from multiple threads.
_recording = False
_recording_lock = threading.Lock()
# FIX #2 (missing Button import): Button and MouseController are now imported
# from pynput.mouse at the top of this file.
_mouse = MouseController()
def _audio_callback(indata, frames, time_info, status):
 with _recording_lock:
 is_recording = _recording
 if is_recording:
 _audio_q.put(indata.copy())
def _flush_audio_queue():
 """FIX #4: Drain any stale audio left from a previous recording cycle."""
 while not _audio_q.empty():
 try:
 _audio_q.get_nowait()
 except queue.Empty:
 break
def _record_audio_i16(stop_event) -> np.ndarray:
 """
 FIX #3 (fatal timing bug): The original code looped 'while _recording',
 but _recording was already False by the time this function was called
 (on_release had already cleared it), so the loop exited instantly with
 zero audio collected.
 Fix: we collect chunks that were placed into _audio_q DURING the key-hold
 by _audio_callback, then drain whatever remains in the queue right now.
 The audio_callback already handles the _recording gate, so we just need
 to pull everything that was queued up.
 """
 chunks = []
 deadline = time.time() + MAX_RECORD_SECONDS
 # Drain everything currently in the queue (the callback filled it while
 # the key was held). Use a short timeout so we don't block forever if
 # the queue is already empty.
 while time.time() < deadline and not stop_event.is_set():
 try:
 chunk = _audio_q.get(timeout=0.05)
 chunks.append(chunk)
 except queue.Empty:
 break # No more chunks waiting — recording is done
 if not chunks:
 return np.zeros((0,), dtype=np.int16)
 audio = np.concatenate(chunks, axis=0).reshape(-1) # float32 [-1, 1]
 return np.clip(audio * 32767, -32768, 32767).astype(np.int16)
def _transcribe(model: Model, audio_i16: np.ndarray) -> str:
 if audio_i16.size == 0:
 return ""
 rec = KaldiRecognizer(model, SAMPLE_RATE)
 rec.AcceptWaveform(audio_i16.tobytes())
 result = json.loads(rec.FinalResult())
 return result.get("text", "").strip().lower()
def _make_message(text: str):
 """
 Convert recognized text -> message dict for main.py to execute.
 Message types main.py will understand:
 TYPE: {"type": "TYPE", "text": "hello"}
 KEY: {"type": "KEY", "key": "space"}
 HOTKEY: {"type": "HOTKEY","keys": ["alt", "left"]}
 SCROLL: {"type": "SCROLL","dir": "down"} (or "up")
 MOUSE: {"type": "MOUSE", "button": "left"} (or "right")
 SEARCH: {"type": "SEARCH","text": "cats"} (type + Enter)
 """
 t = text.strip().lower()
 if not t:
 return None
 # --- Dictation ---
 if t.startswith("type "):
 return {"type": "TYPE", "text": t[5:].strip()}
 if t.startswith("search "):
 phrase = t[7:].strip()
 return {"type": "SEARCH", "text": phrase} if phrase else None
 # --- Media ---
 if t in ("play", "pause", "toggle", "play pause"):
 return {"type": "KEY", "key": "space"}
 # FIX #7 (volume keys): "volumeup" etc. are not valid pyautogui key names
 # on all platforms. We now use pynput Key constants instead, passed as a
 # special PYNPUT_KEY message type that execute_voice_message handles.
 if t in ("volume up", "louder"):
 return {"type": "PYNPUT_KEY", "key": "media_volume_up"}
 if t in ("volume down", "quieter"):
 return {"type": "PYNPUT_KEY", "key": "media_volume_down"}
 if t in ("mute", "unmute"):
 return {"type": "PYNPUT_KEY", "key": "media_volume_mute"}
 # --- Scroll ---
 if t in ("scroll down", "down"):
 return {"type": "SCROLL", "dir": "down"}
 if t in ("scroll up", "up"):
 return {"type": "SCROLL", "dir": "up"}
 # --- Navigation ---
 if t in ("back", "go back"):
 return {"type": "HOTKEY", "keys": ["alt", "left"]}
 # --- Mouse clicks ---
 if t in ("click", "left click"):
 return {"type": "MOUSE", "button": "left"}
 if t == "right click":
 return {"type": "MOUSE", "button": "right"}
 # Fallback: type whatever was said
 return {"type": "TYPE", "text": t}
def voice_worker(out_queue, stop_event):
 """
 Background thread entry point.
 Hold PUSH_TO_TALK_KEY, speak, release → puts one message dict into out_queue.
 """
 global _recording
 model_path = Path(VOSK_MODEL_PATH)
 if not model_path.exists():
 raise FileNotFoundError(
 f"Vosk model folder not found: {model_path}\n"
 "Download a model and set VOSK_MODEL_PATH."
 )
 model = Model(str(model_path))
 def on_press(key):
 global _recording
 if key == PUSH_TO_TALK_KEY:
 with _recording_lock:
 if not _recording:
 _flush_audio_queue() # FIX #4: clear stale audio before new recording
 _recording = True
 def on_release(key):
 global _recording
 if key == PUSH_TO_TALK_KEY:
 with _recording_lock:
 _recording = False
 listener = keyboard.Listener(on_press=on_press, on_release=on_release)
 listener.start()
 try:
 with sd.InputStream(
 samplerate=SAMPLE_RATE,
 channels=CHANNELS,
 dtype="float32",
 callback=_audio_callback,
 blocksize=BLOCKSIZE, # FIX #8: stable fixed blocksize
 ):
 while not stop_event.is_set():
 with _recording_lock:
 currently_recording = _recording
 if currently_recording:
 # Wait until the key is released (recording ends)
 while not stop_event.is_set():
 with _recording_lock:
 if not _recording:
 break
 time.sleep(0.02)
 # FIX #3: Now collect the audio that the callback queued up
 # while the key was held. _recording is already False here,
 # but the chunks are waiting in _audio_q.
 audio_i16 = _record_audio_i16(stop_event)
 text = _transcribe(model, audio_i16)
 msg = _make_message(text)
 if msg:
 out_queue.put(msg)
 time.sleep(0.02)
 finally:
 # FIX #6: always stop the listener so its thread doesn't linger
 listener.stop()
