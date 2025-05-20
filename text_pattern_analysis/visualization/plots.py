"""
Plotting functions for text pattern analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_word_frequency(word_freq, top_n=20, title=None, save_path=None):
    """
    Plot word frequency distribution
    
    Args:
        word_freq (dict or pandas.Series): Word frequency dictionary or Series
        top_n (int): Number of top words to plot
        title (str, optional): Plot title
        save_path (str, optional): Path to save the plot
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    # Convert to Series if it's a dictionary
    if isinstance(word_freq, dict):
        word_freq = pd.Series(word_freq)
    
    # Sort and take top N
    top_words = word_freq.sort_values(ascending=False).head(top_n)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    top_words.plot(kind='bar', ax=ax)
    
    # Set labels and title
    ax.set_ylabel('Frequency')
    ax.set_xlabel('Words')
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'Top {top_n} Word Frequencies')
    
    # Rotate x-labels
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path)
    
    return fig

def plot_pos_distribution(pos_counts, title=None, save_path=None):
    """
    Plot part-of-speech tag distribution
    
    Args:
        pos_counts (dict or pandas.Series): POS tag counts
        title (str, optional): Plot title
        save_path (str, optional): Path to save the plot
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    # Convert to Series if it's a dictionary
    if isinstance(pos_counts, dict):
        pos_counts = pd.Series(pos_counts)
    
    # Sort by value
    pos_counts = pos_counts.sort_values(ascending=False)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    pos_counts.plot(kind='bar', ax=ax)
    
    # Set labels and title
    ax.set_ylabel('Count')
    ax.set_xlabel('Part-of-Speech Tag')
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Part-of-Speech Distribution')
    
    # Rotate x-labels
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path)
    
    return fig

def plot_comparison_bar(human_values, ai_values, labels, title=None, save_path=None):
    """
    Create a comparative bar chart for human vs AI metrics
    
    Args:
        human_values (list): Values for human text
        ai_values (list): Values for AI text
        labels (list): Labels for the x-axis
        title (str, optional): Plot title
        save_path (str, optional): Path to save the plot
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    # Set width of bars
    bar_width = 0.35
    
    # Set position of bars on x axis
    r1 = np.arange(len(human_values))
    r2 = [x + bar_width for x in r1]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(r1, human_values, width=bar_width, label='Human', color='blue', alpha=0.7)
    ax.bar(r2, ai_values, width=bar_width, label='ChatGPT', color='green', alpha=0.7)
    
    # Add labels and title
    if title:
        ax.set_title(title)
    ax.set_xticks([r + bar_width/2 for r in range(len(human_values))])
    ax.set_xticklabels(labels)
    ax.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path)
    
    return fig

def plot_confusion_matrix(cm, class_names=None, title='Confusion Matrix', save_path=None):
    """
    Plot confusion matrix
    
    Args:
        cm (numpy.ndarray): Confusion matrix
        class_names (list, optional): List of class names
        title (str): Plot title
        save_path (str, optional): Path to save the plot
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    if class_names is None:
        class_names = ['Human', 'AI']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    
    # Set labels and title
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title(title)
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path)
    
    return fig 