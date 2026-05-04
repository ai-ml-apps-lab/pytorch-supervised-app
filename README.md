# PyTorch Supervised Learning App 

A lightweight PyTorch app for **classification and regression** with an interactive **Streamlit UI**.
Ideal for quick experiments, demos, and teaching.

---
## Features

Upload CSV datasets

Supports:
    Classification
    Regression

Interactive UI:
    Select target column
    Configure model & hyperparameters

Built-in:
    Training + validation
    Early stopping
    LR scheduler

Visualizations:
    Loss curves
    Accuracy (classification)

Export predictions as CSV

---

## Repository Structure

```
data/
app.py                  # Streamlit UI
pytorch_pipeline.py     # Model + training (OOP)
requirements.txt
README.md
```
---

## Installation

```bash
pip install -r requirements.txt

streamlit run app.py

Open: http://localhost:8501
```

---
## Usage

Upload CSV

Select:
    Task (Classification / Regression)
    Target column
    tune parameters

Click Run Training

View results & download predictions

---
## Key Params

units → hidden layers (e.g. [64,128,64])

activation → relu / tanh / sigmoid / leaky_relu

lr → learning rate

batch_size, epochs

dropout_rate, batch_norm

patience → early stopping
---
## Outputs

1. Classification

    * Accuracy
    * Loss & accuracy curves

2. Regression

   *  MAE / MSE
   * Loss curve

---
## Notes

* units uses eval() (safe locally)

* “r2” in code = actually MSE

* All columns except target are used as features
---

## Deployment
* Local: streamlit run app.py
* Cloud: deploy via Streamlit Cloud (GitHub repo → select app.py)
---