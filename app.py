from flask import Flask, render_template, request
import cv2
import numpy as np
import joblib
import os

app = Flask(__name__)

# ----------------------------------------
# Load trained models
# ----------------------------------------
models = {
    "logistic": joblib.load("models/logistic_regression.pkl"),
    "svm": joblib.load("models/svm.pkl"),
    "rf": joblib.load("models/random_forest.pkl"),
    "kmeans": joblib.load("models/kmeans.pkl"),
}

IMG_SIZE = 64

# ----------------------------------------
# Image preprocessing (SAME as training)
# ----------------------------------------
def preprocess_image(image_path):
    img = cv2.imread(image_path)

    if img is None:
        return None

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.flatten()
    img = img / 255.0

    return img.reshape(1, -1)

# ----------------------------------------
# Routes
# ----------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    model_used = None

    if request.method == "POST":
        file = request.files["image"]
        model_name = request.form["model"]

        if file:
            image_path = "temp.jpg"
            file.save(image_path)

            img = preprocess_image(image_path)
            os.remove(image_path)

            if img is not None:
                model = models[model_name]

                if model_name == "kmeans":
                    cluster = model.predict(img)[0]
                    # cluster → label mapping is implicit here
                    pred = cluster
                else:
                    pred = model.predict(img)[0]

                prediction = "Dog" if pred == 1 else "Cat"
                model_used = model_name.upper()

    return render_template(
        "index.html",
        prediction=prediction,
        model=model_used
    )

# ----------------------------------------
# Run app
# ----------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
