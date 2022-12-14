from flask import Flask, jsonify, request
import whisper
from pathlib import Path
from flask_cors import CORS
import torch
from scipy.io import wavfile
import numpy as np
app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = 'supersecretkey'


@app.route('/',)
def hello():
    return "text"


@app.route('/transcribe', methods=['POST'])
def predict():
    if request.method == 'POST':
        # file_data = request.get_json()['file']
        # decode_file = base64.b64decode(file_data)
        # print(decode_file)
        audio_file = request.data
        with open('w.mp3', 'wb') as f:
            f.write(audio_file)
        # with wavfile.read('Record (online-voice-recorder.com).wav') as (audio, metadata):
        #     n = audio.shape[1]
        #     d = audio.shape[0]
        # with wave.open("Record (online-voice-recorder.com).wav", 'rb') as f:

        #     audio = f.readframes(-1)
        #     audio_data = np.frombuffer(audio, dtype=np.int16)
        #     print(audio_data)
        # sampling_rate, data = wavfile.read(
        #     f'Record (online-voice-recorder.com).wav')
        # with open(f'Record (online-voice-recorder.com) (1).mp3', 'rb') as file:
        #     mp3byte = BytesIO(file.read())
        # mp3 = base64.b64decode(mp3byte.getvalue()).decode("ISO-8859-1")
        # payload = np.frombuffer(mp3, dtype=np.int16)
        # print(payload)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = whisper.load_model("base")
        audio = whisper.load_audio(
            "w.mp3")
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(device)
        _, probs = model.detect_language(mel)
        print(f"Detected language: {max(probs, key=probs.get)}")
        options = whisper.DecodingOptions(fp16=False)
        result = whisper.decode(model, mel, options)
    return jsonify({'Text': result.text, 'language': max(probs, key=probs.get)})


@app.route('/translate', methods=['POST'])
def translate():
    if request.method == 'POST':
        audio_file = request.data
        with open('w.mp3', 'wb') as f:
            f.write(audio_file)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = whisper.load_model("base")
        audio = whisper.load_audio(
            "w.mp3")
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(device)
        _, probs = model.detect_language(mel)
        print(f"Detected language: {max(probs, key=probs.get)}")
        transcribe = model.transcribe(audio, fp16=False, language="en")

        print(transcribe["text"])

    return jsonify({'Text': transcribe["text"]})


@app.route('/detectLanguage', methods=['POST'])
def detectLanguage():
    if request.method == 'POST':
        audio_file = request.data
        with open('w.mp3', 'wb') as f:
            f.write(audio_file)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = whisper.load_model("base")
        audio = whisper.load_audio(
            "w.mp3")
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(device)
        _, probs = model.detect_language(mel)

    return jsonify({'language': max(probs, key=probs.get)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=30006, debug=True)
