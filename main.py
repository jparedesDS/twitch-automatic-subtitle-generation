import os
import pyaudio
from transformers import pipeline
from pydub import AudioSegment
from io import BytesIO

# Configuración de FFmpeg
FFMPEG_PATH = os.getenv("FFMPEG_PATH", r"C:/ffmpeg/bin/ffmpeg.exe")
AudioSegment.converter = FFMPEG_PATH

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2  # Stereo
RATE = 44100  # Frecuencia de muestreo común para audio de salida

# Inicializar modelos de Hugging Face
print("Cargando modelos, por favor espera...")
speech_to_text = pipeline("automatic-speech-recognition", model="openai/whisper-large-v2", framework="pt")
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-es")
print("Modelos cargados correctamente.")

def get_loopback_device_index():
    """Devuelve el índice del dispositivo de audio loopback (Stereo Mix o similar)."""
    p = pyaudio.PyAudio()
    print("Buscando dispositivos de audio...")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if "stereo mix" in info['name'].lower() or "loopback" in info['name'].lower():
            print(f"Dispositivo encontrado: {info['name']} (índice {i})")
            p.terminate()
            return i
    p.terminate()
    raise RuntimeError("No se encontró un dispositivo de loopback (Stereo Mix o similar).")

def capture_and_translate():
    """Captura el audio del sistema en tiempo real y lo traduce de inglés a español."""
    p = pyaudio.PyAudio()
    stream = None

    try:
        device_index = get_loopback_device_index()

        # Abrir el stream de audio desde el dispositivo loopback
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        input_device_index=device_index,
                        frames_per_buffer=CHUNK)

        print("Grabando audio del sistema en tiempo real... (Ctrl+C para detener)")
        audio_buffer = []

        while True:
            try:
                # Leer datos de audio en fragmentos
                data = stream.read(CHUNK, exception_on_overflow=False)
                audio_buffer.append(data)

                # Procesar cada 10 segundos de audio
                if len(audio_buffer) * CHUNK >= RATE * 10:  # 10 segundos
                    print("Procesando fragmento...")

                    # Combinar fragmentos en un único archivo de audio
                    audio_data = b"".join(audio_buffer)
                    audio_buffer = []

                    # Convertir datos de audio en formato de archivo
                    audio_segment = AudioSegment(
                        data=audio_data,
                        sample_width=p.get_sample_size(FORMAT),
                        frame_rate=RATE,
                        channels=CHANNELS
                    )

                    # Guardar el fragmento temporalmente en memoria
                    audio_file = BytesIO()
                    audio_segment.export(audio_file, format="wav")
                    audio_file.seek(0)

                    # Transcribir el audio a texto en inglés
                    transcription = speech_to_text(audio_file)
                    english_text = transcription['text']
                    print(f"Texto en inglés: {english_text}")

                    # Traducir el texto al español
                    translated_text = translator(english_text)[0]['translation_text']
                    print(f"Traducción al español: {translated_text}\n")

            except Exception as e:
                print(f"Error al procesar el fragmento: {e}")

    except KeyboardInterrupt:
        print("\nGrabación detenida.")
    finally:
        print("Cerrando recursos...")
        if stream and stream.is_active():
            stream.stop_stream()
        if stream:
            stream.close()
        p.terminate()

if __name__ == "__main__":
    try:
        capture_and_translate()
    except Exception as e:
        print(f"Error fatal: {e}")
