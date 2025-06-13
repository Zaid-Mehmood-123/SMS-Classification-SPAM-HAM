#📱 SMS Classification: Spam or Ham
A project that classifies SMS messages as Spam or Ham using natural language processing and machine learning models (Naive Bayes, Logistic Regression, SVM, etc.).

Overview

This project builds a machine learning pipeline to classify SMS messages as spam or ham (legitimate). Key steps include:

Data loading

Text preprocessing: normalization, tokenization, stop word removal

Feature extraction: Bag‑of‑Words and TF‑IDF

Model training: Naive Bayes, Logistic Regression, SVM, etc.

Evaluation: Accuracy, Precision, Recall, F1‑score, Confusion Matrix

Comparison of models to select the best performer

📂 Dataset
Source: SMS Spam Collection (5,574 labeled messages: “ham” or “spam”) 

Format: TSV with columns:

v1: label (ham/spam)

v2: raw text message

🚧 Installation & Setup
Ensure you have the following dependencies (e.g. via requirements.txt or direct install):

pip install numpy pandas matplotlib seaborn scikit-learn nltk
If you’d like to use TensorFlow/Keras models later:

pip install tensorflow
Pre-download NLTK resources inside your notebook:

import nltk
nltk.download('stopwords')
nltk.download('punkt')

🧩 Project Structure (Notebook Flow)
Import libraries: pandas, numpy, matplotlib, seaborn, nltk, sklearn

Load dataset and inspect class balance

Text cleaning:

Lowercasing

Removing punctuation & numbers

Tokenization

Stop word removal and optional stemming/lemmatization

Feature engineering:

Convert to numeric features using CountVectorizer and/or TfidfVectorizer

Train/test split

Model training:

Baseline with Multinomial Naive Bayes

Try Logistic Regression, SVM, etc.

Use cross-validation and hyperparameter tuning

Model evaluation:

Metrics: accuracy, precision, recall, F1‑score

Confusion matrix visualization

Select and save the best model

📊 Results & Findings
Naive Bayes with TF-IDF typically achieves above 97% accuracy, aligning with other studies 

Logistic Regression and SVM often perform competitively

Present confusion matrix and detailed metrics in the notebook

🏆 Notebook Highlights
Clean, reproducible pipeline—from raw text to evaluation plots

Comparison across multiple classifiers and vectorization techniques

Insightful visualizations: word clouds, label distribution, confusion matrices

📌 How to Run
Clone or download the notebook from Kaggle.

Upload to Kaggle or your local Jupyter environment.

Ensure all dependencies are installed (see Installation).

Run cells sequentially.

📈 Inspect results and experiment with:

New classifiers (Random Forests, XGBoost)

Advanced text processing (e.g. n‑grams, stemming)

Deep learning via tensorflow.keras

🛠️ Extensions & Ideas
Incorporate word n‑grams, character‑level features

Apply SMOTE or other techniques to balance dataset

Explore deep learning (e.g., LSTM, CNN, Transformer models)

Deploy as a real‑time web API for live SMS filtering

References
SMS Spam Collection Dataset: publicly available on Kaggle 

📄 License
This work is released under an MIT-style license—feel free to use and modify for personal or educational projects!

