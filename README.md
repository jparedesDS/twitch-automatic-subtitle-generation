# Real-Time Audio Translator 🎧🌐

This project captures system audio in real-time, transcribes it into text using a speech recognition model (ASR), and translates it from English to Spanish using Hugging Face models.

## Features
- Real-time audio capture from loopback devices (Stereo Mix or similar).
- Audio-to-text transcription using the `openai/whisper-large-v2` model.
- Transcription translation from English to Spanish using the `Helsinki-NLP/opus-mt-en-es model`.

## Requirements
Python 3.8 or higher.
FFmpeg installed and accessible via the FFMPEG_PATH environment variable.
The following Python packages:
pyaudio
transformers
pydub

- Python 3.8 or higher.
- [FFmpeg](https://ffmpeg.org/) installed and accessible via the `FFMPEG_PATH` environment variable.
- The following Python packages:
  - `pyaudio`
  - `transformers`
  - `pydub`

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/tuusuario/audio-translator.git
   cd audio-translator

2. Install the requirements:

```
pip install -r requirements.txt
```
3. Ensure FFmpeg is installed and properly configured. On Windows, add the FFmpeg path to the FFMPEG_PATH environment variable:
```
set FFMPEG_PATH=C:/ruta/a/ffmpeg/bin/ffmpeg.exe
```
## Usage
1. Run the main script:
```
python main.py
```
2. The program will search for compatible audio devices (Stereo Mix or loopback) and begin capturing system audio.
3. Each audio segment (10 seconds) will be transcribed and translated. The output will be displayed in the terminal.
4. To stop the recording, use Ctrl+C.

## Code Structure
- Audio Capture: The pyaudio library is used to read audio data from a loopback device.
- Audio Processing: The pydub library processes binary data into an audio file format.
- Transcription: The `openai/whisper-large-v2` model converts audio to English text.
- Translation: The transcribed text is translated to Spanish using the `Helsinki-NLP/opus-mt-en-es` model.

## Common Issues
1. Audio device not found:
- `Ensure the loopback or Stereo Mix device is enabled in your system settings.`
2. FFmpeg-related errors:
- `Verify that FFmpeg is installed and that the path is correctly configured.`

## Contributions
Contributions are welcome! If you find a bug or have an idea for improvement, open an issue or submit a pull request.

## License
This project is licensed under the MIT License.

## Credits
Created with ❤️ and Hugging Face models.
