#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main script to run ChatGPT vs Human text pattern analysis
"""

import os
import subprocess
import sys

def check_requirements():
    """Verify that required packages are installed"""
    print("Checking required packages...")
    try:
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import nltk
        import pyspark
        from sklearn import metrics
        print("All required packages are installed.")
    except ImportError as e:
        print(f"Missing required package: {e}")
        print("Please install required packages using: pip install -r requirements.txt")
        sys.exit(1)

def run_notebook(notebook_path):
    """Run a Jupyter notebook using nbconvert"""
    print(f"Running notebook: {notebook_path}")
    try:
        subprocess.run([
            "jupyter", "nbconvert", 
            "--to", "notebook", 
            "--execute", 
            notebook_path
        ], check=True)
        print(f"Successfully executed {notebook_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error executing {notebook_path}: {e}")
        return False
    return True

def main():
    """Main function to run the analyses"""
    check_requirements()
    
    # Download NLTK data
    print("Downloading NLTK data...")
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    
    # List of notebooks to run in order
    notebooks = [
        "codes/Word_frequency.ipynb",
        "codes/Paraphrase_Pattern_GPT.ipynb",
        "codes/Paraphrase_Pattern_Human.ipynb",
        "codes/Paraphrase_SVD.ipynb",
        "Machine_Learning.ipynb"
    ]
    
    # Run notebooks in sequence
    for notebook in notebooks:
        if not os.path.exists(notebook):
            print(f"Warning: Notebook {notebook} not found, skipping.")
            continue
        
        success = run_notebook(notebook)
        if not success:
            print(f"Error running {notebook}, stopping execution.")
            break
    
    print("Analysis complete.")

if __name__ == "__main__":
    main() 