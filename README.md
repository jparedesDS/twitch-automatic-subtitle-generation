# Audio Translator in Real-Time 🎧🌐

Este proyecto permite capturar audio del sistema en tiempo real, transcribirlo a texto utilizando un modelo de reconocimiento de voz (ASR) y traducirlo de inglés a español con ayuda de modelos de Hugging Face. 

## Características

- Captura de audio en tiempo real desde dispositivos loopback (Stereo Mix o similares).
- Transcripción de audio a texto utilizando el modelo `openai/whisper-large-v2`.
- Traducción del texto transcrito de inglés a español con el modelo `Helsinki-NLP/opus-mt-en-es`.

## Requisitos

- Python 3.8 o superior.
- [FFmpeg](https://ffmpeg.org/) instalado y accesible en la variable de entorno `FFMPEG_PATH`.
- Los siguientes paquetes de Python:
  - `pyaudio`
  - `transformers`
  - `pydub`

## Instalación

1. Clona este repositorio:
   ```bash
   git clone https://github.com/tuusuario/audio-translator.git
   cd audio-translator

2. Instala los requisitos:

```
pip install -r requirements.txt
```
3. Asegúrate de que FFmpeg esté instalado y configurado correctamente. En Windows, agrega la ruta de FFmpeg a la variable de entorno FFMPEG_PATH:
```
set FFMPEG_PATH=C:/ruta/a/ffmpeg/bin/ffmpeg.exe
```
## Uso
1. Ejecuta el script principal:
```
python main.py
```
2. El programa buscará dispositivos de audio compatibles (Stereo Mix o loopback) y comenzará a capturar el audio del sistema.
3. Cada fragmento de audio (10 segundos) será transcrito y traducido. La salida se mostrará en la terminal.
4. Para detener la grabación, usa Ctrl+C.

## Estructura del Código
- Captura de Audio: Se utiliza la biblioteca pyaudio para leer datos de audio desde un dispositivo loopback.
- Procesamiento de Audio: Se emplea pydub para convertir los datos binarios en un archivo de audio procesable.
- Transcripción: Utiliza el modelo openai/whisper-large-v2 para convertir el audio a texto en inglés.
- Traducción: Traduce el texto transcrito al español usando el modelo Helsinki-NLP/opus-mt-en-es.
## Problemas Comunes
1. Dispositivo de audio no encontrado:
- `Asegúrate de habilitar el dispositivo loopback o Stereo Mix en la configuración de tu sistema.`
2. Errores relacionados con FFmpeg:
- `Verifica que FFmpeg esté instalado y que la ruta sea correcta.`
 
## Contribuciones
¡Las contribuciones son bienvenidas! Si encuentras un error o tienes una idea para mejorar, abre un issue o envía un pull request.

## Licencia
Este proyecto está licenciado bajo la MIT License.

## Créditos
Creado con ❤️ y los modelos de Hugging Face.
