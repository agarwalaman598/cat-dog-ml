<div align="center">

# 🐱 Cat vs Dog Classifier 🐕

**A Machine Learning web app that classifies images as Cat or Dog**

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Visit_App-blue?style=for-the-badge)](https://agarwalaman.pythonanywhere.com/)
[![Python](https://img.shields.io/badge/Python-3.11-green?style=flat-square&logo=python)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-Web_App-black?style=flat-square&logo=flask)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?style=flat-square&logo=scikit-learn)](https://scikit-learn.org/)

</div>

---

## ✨ Features

- 🖼️ Upload any cat or dog image
- 🤖 Choose from 4 ML models
- ⚡ Instant predictions
- 🎨 Clean, responsive UI

---

## 🚀 Live Demo

👉 **[https://agarwalaman.pythonanywhere.com](https://agarwalaman.pythonanywhere.com/)**

---

## 🧠 Models

| Model | Accuracy |
|:------|:--------:|
| Logistic Regression | ~53% |
| **SVM** | **~60%** |
| Random Forest | ~58% |
| K-Means | ~56% |

---

## 🛠️ Tech Stack

- **Backend:** Flask, Python
- **ML:** scikit-learn, NumPy
- **Image Processing:** OpenCV
- **Frontend:** HTML, CSS

---

## 📦 Local Setup

```bash
# Clone the repo
git clone https://github.com/agarwalaman598/cat-dog-ml.git
cd cat-dog-ml

# Create virtual environment
python -m venv ml_env
ml_env\Scripts\activate  # Windows
source ml_env/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

Open: **http://127.0.0.1:5000**

---

## 📁 Project Structure

```
cat-dog-ml/
├── app.py              # Flask web app
├── preprocess.py       # Image preprocessing
├── train_models.py     # Model training
├── requirements.txt    # Dependencies
├── models/             # Trained models (.pkl)
├── templates/          # HTML templates
└── static/             # CSS styles
```

---

## 👤 Author

**Aman Agarwal**

---

<div align="center">

⭐ Star this repo if you found it helpful!

</div>
