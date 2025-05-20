"""
SVD (Singular Value Decomposition) analysis for text data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

def perform_svd_analysis(texts, n_components=50):
    """
    Perform SVD analysis on text data
    
    Args:
        texts (list): List of text documents
        n_components (int): Number of components for SVD
        
    Returns:
        tuple: (svd_model, X_transformed, vectorizer)
    """
    # Create and fit TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(texts)
    
    # Apply SVD
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_transformed = svd.fit_transform(X)
    
    return svd, X_transformed, vectorizer

def analyze_singular_values(svd_model):
    """
    Analyze the singular values from SVD
    
    Args:
        svd_model: Fitted TruncatedSVD model
        
    Returns:
        dict: Dictionary containing analysis results
    """
    # Get singular values
    singular_values = svd_model.singular_values_
    
    # Calculate explained variance ratio
    explained_variance_ratio = svd_model.explained_variance_ratio_
    cumulative_explained_variance = np.cumsum(explained_variance_ratio)
    
    return {
        "singular_values": singular_values,
        "explained_variance_ratio": explained_variance_ratio,
        "cumulative_explained_variance": cumulative_explained_variance
    }

def plot_singular_values(svd_results, save_path=None):
    """
    Plot the singular values and explained variance
    
    Args:
        svd_results (dict): Results from analyze_singular_values function
        save_path (str, optional): Path to save the plot
        
    Returns:
        tuple: Figure objects (fig1, fig2)
    """
    # Plot singular values
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(svd_results["singular_values"], 'o-')
    ax1.set_title('Singular Values')
    ax1.set_xlabel('Component')
    ax1.set_ylabel('Singular Value')
    ax1.grid(True)
    
    # Plot cumulative explained variance
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(svd_results["cumulative_explained_variance"], 'o-')
    ax2.set_title('Cumulative Explained Variance')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Explained Variance')
    ax2.axhline(y=0.8, color='r', linestyle='-', label='80% Threshold')
    ax2.grid(True)
    ax2.legend()
    
    # Save plots if path is provided
    if save_path:
        fig1.savefig(f"{save_path}/singular_values.png")
        fig2.savefig(f"{save_path}/explained_variance.png")
    
    return fig1, fig2

def get_top_terms_for_components(svd_model, vectorizer, n_terms=10):
    """
    Get the top terms for each SVD component
    
    Args:
        svd_model: Fitted TruncatedSVD model
        vectorizer: Fitted TfidfVectorizer
        n_terms (int): Number of top terms to get per component
        
    Returns:
        list: List of dictionaries containing top terms for each component
    """
    feature_names = np.array(vectorizer.get_feature_names_out())
    components = []
    
    for i, component in enumerate(svd_model.components_):
        # Get indices of top terms for this component
        top_indices = component.argsort()[:-n_terms-1:-1]
        top_terms = feature_names[top_indices]
        top_weights = component[top_indices]
        
        # Store component information
        components.append({
            "component_id": i,
            "top_terms": top_terms,
            "weights": top_weights
        })
    
    return components 