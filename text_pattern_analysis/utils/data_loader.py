"""
Data loading utilities for text pattern analysis
"""

import pandas as pd
import os

def load_paraphrase_dataset(file_path=None):
    """
    Load the ChatGPT Paraphrases dataset
    
    Args:
        file_path (str, optional): Path to the dataset CSV file.
            If None, will check common locations
            
    Returns:
        pandas.DataFrame: Loaded dataset
    """
    # If a specific path is provided, try to load it
    if file_path and os.path.exists(file_path):
        return pd.read_csv(file_path)
    
    # Try common locations for the dataset
    common_paths = [
        "data/paraphrases.csv",
        "paraphrases.csv",
        "dataset/paraphrases.csv"
    ]
    
    for path in common_paths:
        if os.path.exists(path):
            return pd.read_csv(path)
            
    raise FileNotFoundError(
        "Paraphrase dataset not found. Please provide a valid file path."
    )

def split_human_ai_texts(df, text_column="text", category_column="category"):
    """
    Split the dataset into human and AI-generated texts
    
    Args:
        df (pandas.DataFrame): Dataset containing both human and AI texts
        text_column (str): Name of the column containing the text
        category_column (str): Name of the column containing the category 
                              (human or AI)
                              
    Returns:
        tuple: (human_texts, ai_texts) DataFrames
    """
    # Check that required columns exist
    if text_column not in df.columns or category_column not in df.columns:
        raise ValueError(f"Required columns {text_column} or {category_column} not found in the DataFrame")
    
    # Split by category
    human_texts = df[df[category_column] == "human"]
    ai_texts = df[df[category_column] == "ai"]
    
    return human_texts, ai_texts

def save_processed_data(df, output_path, file_name):
    """
    Save processed data to CSV file
    
    Args:
        df (pandas.DataFrame): Data to save
        output_path (str): Directory to save the file in
        file_name (str): Name of the file
        
    Returns:
        str: Path to the saved file
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Create the full path
    full_path = os.path.join(output_path, file_name)
    
    # Save the data
    df.to_csv(full_path, index=False)
    
    return full_path 