"""
Text processing utilities for the text pattern analysis package
"""

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def preprocess_text(text, remove_stopwords=True, lemmatize=True):
    """
    Preprocess text by tokenizing, removing stopwords, and lemmatizing
    
    Args:
        text (str): The input text to preprocess
        remove_stopwords (bool): Whether to remove stopwords
        lemmatize (bool): Whether to lemmatize words
        
    Returns:
        list: List of processed words
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatize
    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return tokens

def get_pos_tags(tokens):
    """
    Get part-of-speech tags for tokens
    
    Args:
        tokens (list): List of word tokens
        
    Returns:
        list: List of (word, pos_tag) tuples
    """
    return nltk.pos_tag(tokens)

def calculate_word_frequencies(tokens):
    """
    Calculate word frequencies from tokens
    
    Args:
        tokens (list): List of word tokens
        
    Returns:
        dict: Dictionary of word frequencies
    """
    freq_dict = {}
    for word in tokens:
        if word in freq_dict:
            freq_dict[word] += 1
        else:
            freq_dict[word] = 1
    return freq_dict 