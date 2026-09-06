import os
import glob
import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib


def extract_features(file):
        audio, sr = librosa.load(file, sr=22050)
        features = []
        #MFCC
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        features.extend(np.mean(mfcc, axis=1))
        features.extend(np.std(mfcc, axis=1))

        #MEL
        mel = librosa.feature.melspectrogram(y=audio, sr=sr)
        mel_db = librosa.power_to_db(mel)
        features.append(np.mean(mel_db))
        features.append(np.std(mel_db))

        #CHROMA
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        features.append(np.mean(chroma)) 
        features.append(np.std(chroma))

        #CONTRAST
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        features.append(np.mean(contrast))   
        features.append(np.std(contrast))

        #ZRC
        zcr = librosa.feature.zero_crossing_rate(audio)
        features.append(np.mean(zcr))
        features.append(np.std(zcr))

        #RMS
        rms = librosa.feature.rms(y=audio)                                      
        features.append(np.mean(rms))
        features.append(np.std(rms))

        return features
X = []
y = []
for file in glob.glob("dataset/real/*.wav"):
    X.append(extract_features(file))
    y.append(0)
for file in glob.glob("dataset/fake/*.wav"):
    X.append(extract_features(file))
    y.append(1)


X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced"
)
model.fit(X_train, y_train)
prediction = model.predict(X_test)
accuracy = accuracy_score(y_test, prediction)
print("Prediction:", prediction)
print("Actual:", y_test)
print("Accuracy:", accuracy)

joblib.dump(model, "model.pkl")



          


        