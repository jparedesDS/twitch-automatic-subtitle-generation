import numpy as np
import sounddevice as sd
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Cargar el procesador y modelo Whisper
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2")

# Parámetros de captura de audio
fs = 16000  # Frecuencia de muestreo

# Función para procesar y transcribir el audio
def transcribir_audio(indata, frames, time, status):
    # Convertir audio a numpy array y procesarlo con Whisper
    audio_input = np.mean(indata, axis=1)  # Convertir a mono (si es estéreo)

    # Procesar el audio para que sea compatible con Whisper
    inputs = processor(audio_input, sampling_rate=fs, return_tensors="pt")

    # Asegurarse de que los datos están en el formato adecuado
    if "input_values" not in inputs:
        print("Error: input_values no encontrado en los datos procesados.")
        return

    # Generar la transcripción
    with torch.no_grad():
        predicted_ids = model.generate(inputs["input_values"])

    # Decodificar la transcripción
    transcription = processor.decode(predicted_ids[0], skip_special_tokens=True)
    print("Transcripción:", transcription)


# Índice del dispositivo de entrada virtual
input_device_index = 18 # Asegúrate de usar el índice correcto

# Abrir el stream para capturar audio
with sd.InputStream(device=input_device_index, channels=1, samplerate=fs, callback=transcribir_audio):
    print("Escuchando el audio de Chrome...")
    sd.sleep(10000)  # Mantener escuchando durante 10 segundos (ajusta el tiempo según sea necesario)