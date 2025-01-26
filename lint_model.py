# prompt: lint a model 

import git
from git import Repo
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

# Install necessary libraries (ensure this is only run once)
# !pip install gitpython pandas scikit-learn mlflow pytest pylint

# Load the dataset
try:
    df = pd.read_csv('/content/drive/MyDrive/Twitter_Data.csv')
except FileNotFoundError:
    print("Error: File not found. Please make sure the file path is correct.")
    exit()

# Data preprocessing (example)
# Assuming the dataset has 'clean_text' and 'category' columns
if {'clean_text', 'category'}.issubset(df.columns):
    df = df[['clean_text', 'category']]
    df = df.dropna()
else:
    print("Error: 'clean_text' or 'category' columns not found in the dataset.")
    exit()

# Feature extraction
vectorizer = TfidfVectorizer(max_features=5000)  # Adjust max_features as needed
X = vectorizer.fit_transform(df['clean_text'])
y = df['category']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set up MLflow experiment
mlflow.set_experiment("Twitter Sentiment Analysis")

with mlflow.start_run():
    # Train a Logistic Regression model
    model = LogisticRegression(max_iter=1000) # Increased max_iter for convergence
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)

    # Log parameters and metrics
    mlflow.log_param("max_features", 5000)
    mlflow.log_metric("accuracy", accuracy)

    # Log the model
    mlflow.sklearn.log_model(model, "twitter_sentiment_model")

    print(f"Accuracy: {accuracy}")
    print(f"Model saved in run {mlflow.active_run().info.run_id}")