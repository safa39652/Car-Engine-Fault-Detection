import os
import numpy as np
import librosa
from flask import Flask, request, render_template_string
import tensorflow as tf

# Charger ton modèle entraîné
model = tf.keras.models.load_model("mon_modele_CNN.keras")

# Prétraitement audio
def preprocess_audio(file_path):
    y, sr = librosa.load(file_path, sr=16000, duration=5.0)

    # Même config que l’entraînement !
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

    max_pad_len = 216
    if mfcc.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0,0),(0,pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_pad_len]

    mfcc = mfcc[np.newaxis, ..., np.newaxis]  # (1, 40, 216, 1)
    return mfcc.astype(np.float32)

# Flask app
app = Flask(__name__)

# Page upload
html_page = """
<!DOCTYPE html>
<html>
<head>
    <title>Détection Problème Moteur</title>
</head>
<body>
    <h2>Uploader un son de moteur :</h2>
    <form action="/predict" method="post" enctype="multipart/form-data">
        <input type="file" name="file" accept=".wav" required>
        <input type="submit" value="Analyser">
    </form>
    {% if prediction %}
        <h3>Résultat : {{ prediction }}</h3>
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(html_page)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "Aucun fichier reçu", 400

    file = request.files["file"]
    filepath = "temp.wav"
    file.save(filepath)

    # Prétraitement
    features = preprocess_audio(filepath)

    # Prédiction
    pred = model.predict(features)
    class_index = np.argmax(pred, axis=1)[0]
    confidence = float(np.max(pred)) * 100   # Pourcentage de confiance

    # ⚠️ Adapter avec tes vraies classes
    classes = [
        "normale",
        "Fuel injector",
        "water_pump",
        "knocking",
        "exhaust_leak",
        "belt_squeal",
        "alternator_pulley"
    ]
    result = classes[class_index]

    # Retour avec prédiction + confiance
    return render_template_string(
        html_page,
        prediction=f"{result} ({confidence:.2f}%)"
    )

if __name__ == "__main__":
    app.run(debug=True)
