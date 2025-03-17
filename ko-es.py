import os
import numpy as np
import pyaudio
from transformers import pipeline
import threading
import time
from queue import Queue
import tkinter as tk
from tkinter import scrolledtext, ttk


"""
8000 Hz (8 kHz)
Más rápido y liviano, pero pierde calidad en la transcripción.
Puede ser suficiente para conversaciones claras y sin ruido.

16000 Hz (16 kHz) ✅ Recomendado
Buen equilibrio entre calidad y rapidez.
Whisper (el modelo que usas) funciona bien con esta tasa.

22050 Hz o 44100 Hz
Mayor calidad, pero más lento en procesamiento.
Puede ser útil si el audio es complejo o hay mucho ruido.
---------------------------------------------------------------------------------------------
CHUNK	maxsize	Latencia	Precisión	Uso recomendado
2048	20-30	🔥 Muy baja	🟡 Media	Respuesta ultrarrápida, pero puede perder contexto.
4096	10-20	⚡ Baja	🟢 Buena	Rápida y con buena precisión, ideal para series en vivo.
5024	5-15	⏳ Media	🟢 Muy buena	Buen equilibrio entre velocidad y contexto.
8192	3-10	🐢 Alta	🟢🔵 Excelente	Traducción más precisa, pero con más retraso.
--------------------------------------------------------------------------------------------
maxsize	Latencia	Precisión	Uso recomendado
5-10	⚡ Muy baja	🟡 Media	Para traducción rápida en tiempo real, con respuesta inmediata.
10-20	⏳ Media	🟢 Buena	Equilibrio entre rapidez y precisión. Ideal para diálogos naturales.
20-30	🐢 Alta	🟢🔵 Muy buena	Si la prioridad es la calidad en textos largos, pero con más retraso.
"""


CHUNK = 4096
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 12000

audio_queue = Queue(maxsize=5)
processing_thread = None
stream = None
p = None
running = False

# Inicializar modelos
tk_root = tk.Tk()
tk_root.title("Transcriptor y Traductor de Audio")
tk_root.attributes('-alpha', 0.9)  # Ajuste de opacidad inicial


def set_opacity(value):
    tk_root.attributes('-alpha', float(value))


def set_font_size(value):
    txt_output.config(font=("Arial", int(value)))


def set_font_color(color):
    txt_output.config(fg=color)


frame_controls = tk.Frame(tk_root)
frame_controls.pack()

opacity_label = tk.Label(frame_controls, text="Opacidad:")
opacity_label.pack(side=tk.LEFT)
opacity_slider = ttk.Scale(frame_controls, from_=0.3, to=1.0, command=set_opacity, orient='horizontal')
opacity_slider.set(0.9)
opacity_slider.pack(side=tk.LEFT)

font_size_label = tk.Label(frame_controls, text="Tamaño de letra:")
font_size_label.pack(side=tk.LEFT)
font_size_slider = ttk.Scale(frame_controls, from_=8, to=24, command=set_font_size, orient='horizontal')
font_size_slider.set(12)
font_size_slider.pack(side=tk.LEFT)

color_label = tk.Label(frame_controls, text="Color de letra:")
color_label.pack(side=tk.LEFT)
color_var = tk.StringVar(value='black')
color_menu = ttk.Combobox(frame_controls, textvariable=color_var, values=['black', 'yellow', 'white'],
                          state='readonly')
color_menu.pack(side=tk.LEFT)
color_menu.bind("<<ComboboxSelected>>", lambda e: set_font_color(color_var.get()))

txt_output = scrolledtext.ScrolledText(tk_root, wrap=tk.WORD, width=60, height=20, font=("Arial", 12))
txt_output.pack()
txt_output.insert(tk.END, "Cargando modelos, por favor espera...\n")
tk_root.update()

speech_to_text = pipeline("automatic-speech-recognition", model="openai/whisper-small", framework="pt")
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-ko-es")
txt_output.insert(tk.END, "Modelos cargados correctamente.\n")
txt_output.update()


def transcribe_and_translate(audio_array):
    try:
        transcription = speech_to_text(audio_array)
        korean_text = transcription['text']
        translated_text = translator(korean_text)[0]['translation_text']
        txt_output.insert(tk.END, f"\nTexto en coreano: {korean_text}\n")
        txt_output.insert(tk.END, f"Traducción al español: {translated_text}\n\n")
        txt_output.see(tk.END)
        txt_output.update()
    except Exception as e:
        txt_output.insert(tk.END, f"Error: {e}\n")


def process_audio():
    while running:
        audio_array = audio_queue.get()
        if audio_array is None:
            break
        transcribe_and_translate(audio_array)


def capture_audio():
    global stream, p, running, processing_thread
    running = True
    p = pyaudio.PyAudio()
    device_index = 0  # Aquí puedes modificar para seleccionar otro dispositivo

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    input_device_index=device_index,
                    frames_per_buffer=CHUNK)

    txt_output.insert(tk.END, "Grabando audio en tiempo real...\n")
    txt_output.update()

    processing_thread = threading.Thread(target=process_audio)
    processing_thread.start()

    audio_buffer = []
    while running:
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_buffer.append(data)
            if len(audio_buffer) * CHUNK >= RATE * 3:
                audio_data = b"".join(audio_buffer)
                audio_buffer = []
                audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                if not audio_queue.full():
                    audio_queue.put(audio_array)
        except Exception as e:
            txt_output.insert(tk.END, f"Error: {e}\n")
            break


def start_recording():
    threading.Thread(target=capture_audio, daemon=True).start()


def stop_recording():
    global running, stream, p
    running = False
    if stream and stream.is_active():
        stream.stop_stream()
        stream.close()
    if p:
        p.terminate()
    txt_output.insert(tk.END, "\nGrabación detenida.\n")
    txt_output.update()


tk.Button(tk_root, text="Iniciar Grabación", command=start_recording).pack()
tk.Button(tk_root, text="Detener Grabación", command=stop_recording).pack()

tk_root.mainloop()
