"""
Machine learning classifiers for human vs AI text detection
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

def prepare_data_for_classification(texts, labels, test_size=0.2, random_state=42):
    """
    Prepare text data for classification by converting to TF-IDF features
    
    Args:
        texts (list): List of text documents
        labels (list): List of labels corresponding to the texts
        test_size (float): Proportion of the dataset to include in the test split
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, vectorizer)
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=random_state
    )
    
    # Create and fit TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    return X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """
    Train and evaluate multiple classification models
    
    Args:
        X_train: Training features
        X_test: Testing features
        y_train: Training labels
        y_test: Testing labels
        
    Returns:
        dict: Dictionary of model results
    """
    # Define models to train
    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM": SVC(probability=True),
        "Random Forest": RandomForestClassifier()
    }
    
    results = {}
    
    # Train and evaluate each model
    for name, model in models.items():
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = metrics.accuracy_score(y_test, y_pred)
        precision = metrics.precision_score(y_test, y_pred, pos_label='ai')
        recall = metrics.recall_score(y_test, y_pred, pos_label='ai')
        f1 = metrics.f1_score(y_test, y_pred, pos_label='ai')
        
        # Store results
        results[name] = {
            "model": model,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": metrics.confusion_matrix(y_test, y_pred)
        }
    
    return results

def get_best_model(results):
    """
    Get the best performing model based on F1 score
    
    Args:
        results (dict): Dictionary of model results from train_and_evaluate_models
        
    Returns:
        tuple: (best_model_name, best_model_object, best_f1_score)
    """
    best_model = None
    best_score = -1
    best_name = None
    
    for name, result in results.items():
        if result["f1"] > best_score:
            best_score = result["f1"]
            best_model = result["model"]
            best_name = name
    
    return best_name, best_model, best_score 