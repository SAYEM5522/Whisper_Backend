import youtube_dl
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


@app.route('/Transcribe', methods=['POST'])
def predict():
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
        options = whisper.DecodingOptions(fp16=False)
        result = whisper.decode(model, mel, options)
    return jsonify({'Text': result.text, 'language': max(probs, key=probs.get)})


@app.route('/Translate', methods=['POST'])
def translate():
    if request.method == 'POST':
        audio_file = request.data
        with open('w.mp3', 'wb') as f:
            f.write(audio_file)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = whisper.load_model("small")
        audio = whisper.load_audio(
            "w.mp3")
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(device)
        _, probs = model.detect_language(mel)
        print(f"Detected language: {max(probs, key=probs.get)}")
        transcribe = model.transcribe(audio, fp16=False, language="en")
        print(transcribe["text"])

    return jsonify({'Text': transcribe["text"]})


@app.route('/Detect Language', methods=['POST'])
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
        print(max(probs, key=probs.get))

    return jsonify({'language': max(probs, key=probs.get)})


@app.route('/Video', methods=['POST'])
def run():
    value = request.args['value']
    url = request.form["url"]
    video_info = youtube_dl.YoutubeDL().extract_info(
        url=url, download=False
    )
    filename = f"{video_info['title']}.mp3"
    options = {
        'format': 'bestaudio/best',
        'keepvideo': False,
        'outtmpl': filename,
    }

    with youtube_dl.YoutubeDL(options) as ydl:
        ydl.download([video_info['webpage_url']])
    if value == 'Transcribe':
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = whisper.load_model("base")
        audio = whisper.load_audio(
            f"{video_info['title']}.mp3")
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(device)
        _, probs = model.detect_language(mel)
        print(f"Detected language: {max(probs, key=probs.get)}")
        options = whisper.DecodingOptions(fp16=False)
        result = whisper.decode(model, mel, options)
        print(result.text)
        return jsonify({'Text': result.text, 'language': max(probs, key=probs.get)})
    if value == 'Translate':
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = whisper.load_model("base")
        audio = whisper.load_audio(
            f"{video_info['title']}.mp3")
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(device)
        _, probs = model.detect_language(mel)
        print(f"Detected language: {max(probs, key=probs.get)}")
        transcribe = model.transcribe(audio, fp16=False, language="en")
        print(transcribe["text"])
        return jsonify({'Text': transcribe["text"]})

    return None


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=30006, debug=True)
