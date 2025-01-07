import pyaudio
import wave
from transformers import pipeline
from pydub import AudioSegment
from io import BytesIO

# Inicializa los modelos
speech_to_text = pipeline("automatic-speech-recognition", model="openai/whisper-medium")
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-es")

# Configuración de PyAudio
CHUNK = 1024  # Tamaño de fragmento
FORMAT = pyaudio.paInt16  # Formato de audio
CHANNELS = 1  # Mono
RATE = 16000  # Frecuencia de muestreo


# Función para capturar y procesar audio en tiempo real
def capture_and_translate():
    # Inicializa PyAudio
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("Grabando audio en tiempo real... (presiona Ctrl+C para detener)")
    try:
        audio_buffer = []  # Para almacenar datos de audio

        while True:
            # Captura audio en fragmentos
            data = stream.read(CHUNK)
            audio_buffer.append(data)

            # Procesar cada 5 segundos de audio
            if len(audio_buffer) * CHUNK >= RATE * 5:  # 5 segundos de audio
                print("Procesando fragmento...")

                # Combina el audio capturado
                audio_data = b"".join(audio_buffer)
                audio_buffer = []  # Limpia el buffer

                # Convierte los datos de audio a un formato de archivo
                audio_segment = AudioSegment(
                    data=audio_data,
                    sample_width=p.get_sample_size(FORMAT),
                    frame_rate=RATE,
                    channels=CHANNELS
                )

                # Guarda temporalmente el fragmento en memoria
                audio_file = BytesIO()
                audio_segment.export(audio_file, format="wav")
                audio_file.seek(0)

                # Transcribe el audio a texto en inglés
                transcription = speech_to_text(audio_file)
                english_text = transcription['text']
                print(f"Texto en inglés: {english_text}")

                # Traduce el texto al español
                translated_text = translator(english_text)[0]['translation_text']
                print(f"Traducción al español: {translated_text}")

    except KeyboardInterrupt:
        print("\nDeteniendo la grabación.")
    finally:
        # Cierra el stream y PyAudio
        stream.stop_stream()
        stream.close()
        p.terminate()


# Ejecuta la función
capture_and_translate()
