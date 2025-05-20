# Utility Scripts

This document describes utility scripts added to help run the analysis more efficiently.

## fix_typos.py

This script automatically fixes common typos in Jupyter notebook files:
- Corrects `totla_paraphrases` to `total_paraphrases`

### Usage:
```
python fix_typos.py
```

## run_analysis.py

This script automates running all analysis notebooks in sequence:
1. Checks required dependencies are installed
2. Downloads NLTK data
3. Runs each notebook in the correct order

### Usage:
```
python run_analysis.py
```

### Notebook Execution Order:
1. `codes/Word_frequency.ipynb`
2. `codes/Paraphrase_Pattern_GPT.ipynb` 
3. `codes/Paraphrase_Pattern_Human.ipynb`
4. `codes/Paraphrase_SVD.ipynb`
5. `Machine_Learning.ipynb`

## requirements.txt

Contains all the Python package dependencies needed to run the analyses.

### Installation:
```
pip install -r requirements.txt
``` 