# 📧 Email Spam Classifier  

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/scikit--learn-ML-yellow?style=for-the-badge&logo=scikit-learn" />
  <img src="https://img.shields.io/badge/NLP-Text%20Processing-purple?style=for-the-badge&logo=nlp" />
  <img src="https://img.shields.io/badge/License-MIT-red?style=for-the-badge" />
</p>

<p align="center">
  A machine learning project that classifies <b>emails/SMS as spam or not spam</b> using NLP techniques.  
  Built with <b>Count Vectorizer, Logistic Regression, and MultinomialNB</b>, trained on the <b>SMS Spam Collection dataset</b>.  
</p>

---

## 🧠 About
This project leverages the **SMS Spam Collection dataset** to build an effective spam detection system.  
- Preprocessing and vectorization of text were performed using **Count Vectorizer**.  
- Classification was implemented with **Logistic Regression** and **Multinomial Naive Bayes (MultinomialNB)**.  
- The system demonstrates high accuracy in spam detection and can handle **custom input predictions**.  

It highlights the power of **natural language processing (NLP)** and **machine learning** in solving real-world problems.  

---

## ✨ Features
- 📩 Classifies emails/SMS as **Spam** or **Not Spam**  
- 🔤 Uses **Count Vectorizer** for text feature extraction  
- 🤖 Models trained with **Logistic Regression** and **MultinomialNB**  
- 💾 Pre-trained model and vectorizer stored in `.pkl` files  
- 🖥️ Flask app (`app.py`) for running predictions interactively  

---

## 📊 Dataset
- **Dataset:** [SMS Spam Collection dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)  
- Format: `spam.csv`  
- Contains **5,572 labeled messages** (`spam` or `ham`)  
- Target variable: `label` (spam/ham)  

---

## ⚙️ Installation

```bash
# Clone repo
git clone https://github.com/Danakin01/email_classifier.git
cd email_classifier

# (Optional) Create virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

🚀 Usage
Run Jupyter Notebook
``` bash

jupyter notebook model_implementation.ipynb
Train & Save Model
The notebook trains the model and generates:

model.pkl → Trained Logistic Regression / NB model

vectorizer.pkl → Count Vectorizer
``` 
Run the Flask App
```
python app.py
```
Then open your browser at: http://127.0.0.1:5000/

Enter a custom message → The app predicts Spam or Not Spam.

🗂️ Project Structure
```bash

email_classifier/
│── app.py                     # Flask app for predictions
│── model.pkl                  # Saved ML model
│── vectorizer.pkl             # Saved Count Vectorizer
│── model_implementation.ipynb # Model training & evaluation
│── spam.csv                   # SMS Spam dataset
└── README.md
```

🛠️ Dependencies
``` Package	Purpose
pandas	Data analysis
numpy	Numerical operations
scikit-learn	ML algorithms & evaluation
flask	Web app for predictions
nltk	Text preprocessing
joblib	Saving/loading models
```
📝 Notes
Logistic Regression provides strong accuracy on this dataset.

MultinomialNB is effective for word-based predictions.

Preprocessing steps (lowercasing, stopwords removal, vectorization) are crucial for model performance.

Extendable to email datasets or other spam detection tasks.

📜 License
This project is licensed under the MIT License.

🤝 Contributing
Fork the repo

Create a feature branch (git checkout -b feature/YourFeature)

Commit changes (git commit -m "Add feature")

Push (git push origin feature/YourFeature)

Open a Pull Request

📧 Contact
💡 Issues & PRs welcome!

📩 Email: danielakinwande00@gmail.com
🌐 GitHub: Danakin01
🔗 LinkedIn: Daniel Akinwande

<p align="center">Built with ❤️ by <b>DANAKIN</b></p> 
