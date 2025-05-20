#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to fix typos in Jupyter notebooks
"""

import json
import os
import sys

def fix_notebook_typos(notebook_path):
    """Fix typos in a Jupyter notebook"""
    try:
        # Read the notebook file
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        # Track if any changes were made
        changes_made = False
        
        # Process each cell
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                # Fix typos in code cells
                for i, source_line in enumerate(cell['source']):
                    # Fix the "totla_paraphrases" typo
                    if 'totla_paraphrases' in source_line:
                        cell['source'][i] = source_line.replace('totla_paraphrases', 'total_paraphrases')
                        changes_made = True
        
        # Write the corrected notebook back
        if changes_made:
            with open(notebook_path, 'w', encoding='utf-8') as f:
                json.dump(notebook, f, indent=1)
            print(f"Fixed typos in {notebook_path}")
        else:
            print(f"No typos found in {notebook_path}")
        
        return changes_made
    
    except Exception as e:
        print(f"Error processing {notebook_path}: {e}")
        return False

def main():
    """Main function to fix typos in notebooks"""
    # List of notebooks to process
    notebooks = [
        "codes/Word_frequency.ipynb",
        "codes/Paraphrase_Pattern_GPT.ipynb",
        "codes/Paraphrase_Pattern_Human.ipynb",
        "codes/Paraphrase_SVD.ipynb",
        "Machine_Learning.ipynb"
    ]
    
    # Process each notebook
    fixed_count = 0
    for notebook in notebooks:
        if os.path.exists(notebook):
            if fix_notebook_typos(notebook):
                fixed_count += 1
        else:
            print(f"Warning: Notebook {notebook} not found, skipping.")
    
    print(f"Fixed typos in {fixed_count} notebook(s).")

if __name__ == "__main__":
    main() 