# Getting Started

This guide will help you get up and running with the ChatGPT vs Human Text Analysis project.

## Prerequisites

1. Python 3.6 or later
2. PySpark (requires Java 8 or later)
3. Jupyter Notebook

## Installation

### Option 1: Using pip

1. Clone the repository:
   ```bash
   git clone https://github.com/pileuszu/text-analysis-ai-vs-human.git
   cd data_mining_team13
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Option 2: Development Installation

For developers who want to modify the code:

1. Clone the repository:
   ```bash
   git clone https://github.com/pileuszu/text-analysis-ai-vs-human.git
   cd data_mining_team13
   ```

2. Install in development mode:
   ```bash
   pip install -e .
   ```

## Download Dataset

1. Download the "ChatGPT Paraphrases" dataset from Kaggle
2. Place the dataset file in the project root directory or in a `/data` subdirectory

## Running the Analysis

### Option 1: Using the Automated Script

Run all analyses in sequence:

```bash
python run_analysis.py
```

This will:
1. Check for required dependencies
2. Download NLTK data
3. Run each notebook in the correct order

### Option 2: Running Individual Notebooks

Open and run the Jupyter notebooks in the following order:

1. `codes/Word_frequency.ipynb`
2. `codes/Paraphrase_Pattern_GPT.ipynb`
3. `codes/Paraphrase_Pattern_Human.ipynb`
4. `codes/Paraphrase_SVD.ipynb`
5. `Machine_Learning.ipynb`

## Using the Package Components

The `text_pattern_analysis` package provides reusable functions for text analysis:

```python
from text_pattern_analysis.utils.text_processing import preprocess_text
from text_pattern_analysis.utils.data_loader import load_paraphrase_dataset
from text_pattern_analysis.analysis.frequent_patterns import mine_frequent_patterns
from text_pattern_analysis.visualization.plots import plot_word_frequency

# Load dataset
df = load_paraphrase_dataset()

# Preprocess text
processed_text = preprocess_text("Sample text to analyze")

# Plot word frequencies
plot_word_frequency(word_freq_dict, top_n=20)
```

## Output Files

Analysis results will be saved in the `dataming_result/` directory, including:
- Word frequency distributions
- Frequent patterns and association rules
- SVD analysis results
- Machine learning model performance metrics

## Fixing Typos in Notebooks

If you encounter typo issues in the notebooks, run:

```bash
python fix_typos.py
```

This will automatically correct common typos in the notebook files. 