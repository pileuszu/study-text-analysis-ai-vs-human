from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="text_pattern_analysis",
    version="0.1.0",
    author="Team 13",
    author_email="team13@example.com",
    description="Analyze and compare text patterns between human-written and ChatGPT-generated content",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/data_mining_team13",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "pyspark>=3.3.0",
        "nltk>=3.8.0",
        "scikit-learn>=1.2.0",
        "jupyter>=1.0.0",
    ],
    entry_points={
        "console_scripts": [
            "run-text-analysis=run_analysis:main",
            "fix-notebook-typos=fix_typos:main",
        ],
    },
) 