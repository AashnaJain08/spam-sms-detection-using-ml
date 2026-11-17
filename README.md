# Spam SMS Detection Using Machine Learning

A machine learning-based web application that classifies SMS messages as **Spam** or **Ham (Normal)** using Natural Language Processing techniques.

## Project Overview

This project allows users to enter an SMS message and check whether it is likely to be spam or a normal message.

The application uses a Logistic Regression model and TF-IDF vectorization for text classification.

## Features

- SMS spam and ham classification
- Interactive Streamlit interface
- TF-IDF text vectorization
- Prediction probability display
- Easy-to-use message input

## Technologies Used

- Python
- Machine Learning
- Natural Language Processing (NLP)
- Scikit-learn
- TF-IDF
- Logistic Regression
- Streamlit
- Joblib

## Model Information

- Algorithm: Logistic Regression
- Text Vectorization: TF-IDF
- Features: Unigrams and Bigrams
- Accuracy: 97.49% (as reported in the application)

## Project Structure

```text
spam-sms-detection-using-ml/
├── app.py
├── requirements.txt
├── spam_model.pkl
├── tfidf_vectorizer.pkl
├── data/
└── notebooks/
```

## How to Run the Project

1. Clone the repository.
2. Open the project folder.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the application:

```bash
streamlit run app.py
```

## How It Works

1. User enters an SMS message.
2. TF-IDF converts the text into numerical features.
3. Logistic Regression predicts the message category.
4. The application displays Spam or Ham with prediction probabilities.

## Project Type

Academic Machine Learning Project

**Note:** This application provides automated predictions and should not be treated as a guarantee that a message is safe.
