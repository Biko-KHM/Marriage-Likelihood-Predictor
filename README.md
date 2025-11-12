# 💍 Marriage Likelihood Prediction AI

> **Disclaimer:** Fate and love are ultimately determined by **God**. This project is educational and explores how AI can analyze patterns of compatibility and relationship dynamics. It is **not** a verdict on human relationships and must be used responsibly.

## Overview
This repository demonstrates the AI development workflow applied to a creative real-world problem: predicting the likelihood that a couple will marry based on quantified relationship features. It follows the CRISP-DM framework and includes ethical considerations, model evaluation, and reproducible notebooks.

## Project structure
```
ai-marriage-likelihood/
├── data/
│   ├── raw/
│   └── processed/             # marriage_data.csv
├── models/                    # saved model artifact (marriage_model.pkl)
├── notebooks/
│   ├── 00-generate-data.ipynb
│   ├── 01-data-exploration.ipynb
│   ├── 02-model-training.ipynb
│   └── 03-evaluation.ipynb
├── src/
│   ├── data/preprocess.py
│   ├── models/train_model.py
│   ├── eval/evaluate.py
│   └── app/app.py
├── reports/
│   └── final_report.md
├── requirements.txt
├── Dockerfile
└── README.md
```

## Installation (local)
```bash
git clone https://github.com/<your-username>/ai-marriage-likelihood.git
cd ai-marriage-likelihood
python -m venv venv
# Activate:
# Windows (Git Bash): source venv/Scripts/activate
# Windows (PowerShell): .
env\Scripts\Activate.ps1
# macOS / Linux: source venv/bin/activate
pip install -r requirements.txt
```

## Usage (quick)
1. Generate synthetic data: open and run `notebooks/00-generate-data.ipynb`.  
2. Explore data: run `notebooks/01-data-exploration.ipynb`.  
3. Train model: run `notebooks/02-model-training.ipynb` or `python src/models/train_model.py`.  
4. Evaluate: run `notebooks/03-evaluation.ipynb` or `python src/eval/evaluate.py`.  
5. Sample prediction: `python src/app/app.py` (loads `models/marriage_model.pkl`).

## Evaluation snapshot
Example classification results from a final test run (see `notebooks/03-evaluation`):
```
accuracy: 0.52
precision (low): 0.50, recall (low): 0.90
precision (high): 0.62, recall (high): 0.16
```
Interpretation: model is good at detecting couples unlikely to marry but weak at identifying true positives; further data balancing and richer features are recommended.

## Ethics & Responsible Use
- Use only anonymized, consensual data.  
- Clearly explain limitations and uncertainties to users.  
- Avoid making life-changing decisions solely based on model outputs.  
- Include opt-out and human review mechanisms in any future product.

## Author
Bikila Keneni — AI for Software Engineering (2025)

---
