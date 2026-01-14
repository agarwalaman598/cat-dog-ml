# Cat vs Dog Image Classification using Machine Learning

## 📌 Objective
To classify images as **Cat** or **Dog** using traditional **Machine Learning algorithms** and deploy the trained models using a **Flask web application**.

This project is implemented as part of a Machine Learning laboratory experiment and avoids deep learning to demonstrate classical ML techniques on image data.

---

## 🧠 Models Used
- Logistic Regression  
- Support Vector Machine (SVM)  
- Random Forest  
- K-Means Clustering (unsupervised)

---

## 📂 Project Structure
```
cat-dog-ml/
├── app.py
├── preprocess.py
├── train_models.py
├── requirements.txt
├── README.md
├── models/
│   ├── logistic_regression.pkl
│   ├── svm.pkl
│   ├── random_forest.pkl
│   └── kmeans.pkl
├── templates/
│   └── index.html
└── static/
    └── style.css
```

---

## 🗂 Dataset
- Dataset: Dogs vs Cats (Kaggle)
- Structure:
```
dataset/
├── cats/
└── dogs/
```

Dataset is not included in this repository due to size limitations.

---

## ⚙️ Installation & Setup

### Create Virtual Environment
```bash
python -m venv ml_env
ml_env\Scripts\activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🔄 How to Run

### Preprocess Images
```bash
python preprocess.py
```

### Train Models
```bash
python train_models.py
```

### Run Flask App
```bash
python app.py
```

Open browser:
```
http://127.0.0.1:5000
```

---

## 📊 Results (Approx. Accuracy)

| Model | Accuracy |
|------|---------|
| Logistic Regression | ~53% |
| SVM | ~60% |
| Random Forest | ~58% |
| K-Means | ~56% |

Best model: **SVM**

---

## 📘 Key Learnings
- Image preprocessing for ML
- Supervised vs unsupervised learning
- Model comparison
- ML deployment with Flask

---

## 👤 Author
Aman Agarwal  
B.Tech CSE
