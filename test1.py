import os
import numpy as np
import pyaudio
from pydub import AudioSegment
from transformers import pipeline
import threading
import time
from queue import Queue

# Configuración de FFmpeg
FFMPEG_PATH = os.getenv("FFMPEG_PATH", r"C:/ffmpeg/bin/ffmpeg.exe")
AudioSegment.converter = FFMPEG_PATH

CHUNK = 1024  # Tamaño de la captura de audio
FORMAT = pyaudio.paInt16
CHANNELS = 1  # Forzar audio mono
RATE = 16000  # Reducción de la frecuencia de muestreo para mejorar rendimiento (16kHz es suficiente)

# Inicializar modelos de Hugging Face
print("Cargando modelos, por favor espera...")
speech_to_text = pipeline("automatic-speech-recognition", model="openai/whisper-small", framework="pt")
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-es")
print("Modelos cargados correctamente.")

def list_audio_devices():
    """Lista los dispositivos de audio disponibles."""
    p = pyaudio.PyAudio()
    for i in range(p.get_device_count()):
        device_info = p.get_device_info_by_index(i)
        print(f"Dispositivo {i}: {device_info['name']}")
    p.terminate()

def get_loopback_device_index():
    """Devuelve el índice de un dispositivo loopback configurado manualmente o seleccionado dinámicamente."""
    print("Lista de dispositivos disponibles:")
    list_audio_devices()
    device_index = 0 #int(input("Selecciona el índice del dispositivo de entrada: "))
    print(f"Usando el dispositivo de entrada con índice {device_index}.")
    return device_index

def audiosegment_to_numpy(audio_segment):
    """Convierte un AudioSegment en un array de NumPy."""
    samples = np.array(audio_segment.get_array_of_samples())
    return samples

def transcribe_and_translate(audio_array):
    """Transcribe y traduce fragmentos de audio."""
    try:
        # Transcribir el audio a texto en inglés
        transcription = speech_to_text(audio_array)
        english_text = transcription['text']
        print(f"Texto en inglés: {english_text}")

        # Traducir el texto al español
        translated_text = translator(english_text)[0]['translation_text']
        print(f"Traducción al español: {translated_text}\n")
    except Exception as e:
        print(f"Error al procesar el fragmento: {e}")


def process_audio(queue):
    """Procesa fragmentos de audio desde la cola."""
    while True:
        audio_array = queue.get()
        if audio_array is None:  # Señal de detención
            break
        transcribe_and_translate(audio_array)


def capture_and_translate():
    """Captura audio y lo traduce en tiempo real."""
    p = pyaudio.PyAudio()
    stream = None
    audio_queue = Queue(maxsize=10)  # Limitar el tamaño de la cola

    try:
        device_index = get_loopback_device_index()

        # Abrir stream de audio
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        input_device_index=device_index,
                        frames_per_buffer=CHUNK)

        print("Grabando audio en tiempo real...")
        audio_buffer = []
        processing_thread = threading.Thread(target=process_audio, args=(audio_queue,))
        processing_thread.start()

        while True:
            try:
                # Capturar datos de audio
                data = stream.read(CHUNK, exception_on_overflow=False)
                audio_buffer.append(data)

                # Procesar cada 1 segundo de audio
                if len(audio_buffer) * CHUNK >= RATE*3:
                    # Combinar fragmentos y procesar
                    audio_data = b"".join(audio_buffer)
                    audio_buffer = []

                    audio_segment = AudioSegment(
                        data=audio_data,
                        sample_width=p.get_sample_size(FORMAT),
                        frame_rate=RATE,
                        channels=CHANNELS
                    )

                    audio_array = audiosegment_to_numpy(audio_segment).astype(np.float32) / 32768.0

                    # Agregar a la cola para procesar
                    if not audio_queue.full():
                        audio_queue.put(audio_array)
                    else:
                        print("Cola llena, descartando fragmento.")

            except Exception as e:
                print(f"Error al capturar audio: {e}")

    except KeyboardInterrupt:
        print("\nGrabación detenida.")
    finally:
        print("Cerrando recursos...")
        if stream and stream.is_active():
            stream.stop_stream()
        if stream:
            stream.close()
        p.terminate()

        # Detener el hilo
        audio_queue.put(None)
        processing_thread.join()

if __name__ == "__main__":
    try:
        capture_and_translate()
    except Exception as e:
        print(f"Error fatal: {e}")