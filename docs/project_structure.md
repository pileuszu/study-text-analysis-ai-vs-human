# Project Structure

This document describes the organization of the code and data in this project.

## Directory Structure

```
data_mining_team13/
├── README.md                 # Project overview and instructions
├── SCRIPTS.md                # Documentation for utility scripts
├── requirements.txt          # Python package dependencies
├── setup.py                  # Package installation configuration
├── run_analysis.py           # Script to run all analyses
├── fix_typos.py              # Script to fix common typos in notebooks
├── Machine_Learning.ipynb    # Machine learning analysis notebook
├── codes/                    # Jupyter notebooks for analysis
│   ├── Word_frequency.ipynb  # Word frequency analysis
│   ├── Paraphrase_Pattern_GPT.ipynb   # GPT text pattern analysis
│   ├── Paraphrase_Pattern_Human.ipynb # Human text pattern analysis
│   └── Paraphrase_SVD.ipynb  # SVD analysis
├── dataming_result/          # CSV files with analysis results
└── text_pattern_analysis/    # Python package with reusable components
    ├── __init__.py
    ├── utils/                # Utility functions
    │   ├── __init__.py
    │   ├── text_processing.py # Text preprocessing functions
    │   └── data_loader.py    # Functions to load and process data
    ├── analysis/             # Analysis modules
    │   ├── __init__.py
    │   ├── frequent_patterns.py # Frequent pattern mining
    │   └── svd_analysis.py   # SVD analysis functions
    ├── models/               # Machine learning models
    │   ├── __init__.py
    │   └── classifiers.py    # Text classification models
    └── visualization/        # Visualization tools
        ├── __init__.py
        └── plots.py          # Plotting functions
```

## Code Organization

The project is organized as follows:

1. **Jupyter Notebooks** - The main analysis is contained in Jupyter notebooks in the `codes/` directory and the root directory.

2. **Python Package** - The `text_pattern_analysis` package contains reusable components extracted from the notebooks:
   - `utils`: Common utility functions for text processing and data loading
   - `analysis`: Modules for performing various analyses (frequent patterns, SVD)
   - `models`: Machine learning models for text classification
   - `visualization`: Plotting and visualization functions

3. **Utility Scripts** - Helper scripts to automate common tasks:
   - `run_analysis.py`: Runs all notebooks in sequence
   - `fix_typos.py`: Fixes common typos in the notebooks

4. **Configuration Files**:
   - `requirements.txt`: Lists all required Python packages
   - `setup.py`: Package installation configuration

## Data Flow

The data flow in this project follows these steps:

1. Load the paraphrase dataset containing human and AI texts
2. Preprocess the texts (tokenization, stop word removal, etc.)
3. Perform various analyses:
   - Word frequency analysis
   - Frequent pattern mining using FP-Growth algorithm
   - SVD analysis to understand text structure
4. Train machine learning models to classify texts
5. Visualize and interpret the results 