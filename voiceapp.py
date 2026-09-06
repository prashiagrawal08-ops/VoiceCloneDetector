from flask import Flask, render_template, request
import librosa
import numpy as np
import joblib
import os

app = Flask(__name__)

model = joblib.load("model.pkl")


def extract_features(file):

    audio, sr = librosa.load(file, sr=22050)

    features = []

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))

    mel = librosa.feature.melspectrogram(y=audio, sr=sr)
    mel = librosa.power_to_db(mel)
    features.append(np.mean(mel))
    features.append(np.std(mel))

    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    features.append(np.mean(chroma))
    features.append(np.std(chroma))

    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    features.append(np.mean(contrast))
    features.append(np.std(contrast))

    zcr = librosa.feature.zero_crossing_rate(audio)
    features.append(np.mean(zcr))
    features.append(np.std(zcr))

    rms = librosa.feature.rms(y=audio)
    features.append(np.mean(rms))
    features.append(np.std(rms))

    return features


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["audio"]

    file.save("temp.wav")

    features = extract_features("temp.wav")

    prediction = model.predict([features])[0]

    probability = model.predict_proba([features])[0]
    confidence = max(probability) * 100

    os.remove("temp.wav")

    if prediction == 1:
        result = "AI GENERATED"
    else:
        result = "REAL VOICE"

    return render_template(
        "index.html",
        result=result,
        confidence=round(confidence, 2)
    )
app.run(host="0.0.0.0", port=10000)


