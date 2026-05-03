🧠 PyTorch Supervised Learning App (Streamlit UI)

A simple and flexible PyTorch-based app for classification and regression, wrapped in an interactive Streamlit UI.
Built for quick experiments, teaching, and lightweight ML workflows.

🚀 Features
Upload your own CSV dataset
Supports:
✅ Classification
✅ Regression
Interactive UI:
Select target column
Configure model architecture
Tune hyperparameters
Built-in:
Training & validation loop
Early stopping
Learning rate scheduler
Visualizations:
Loss curves
Accuracy (classification)
Regression scatter plots
Export:
📥 Download predictions as CSV
📁 Project Structure
.
├── data/
│   ├── housing.csv
│   └── iris.csv
├── app.py                  # Streamlit UI
├── pytorch_pipeline.py     # Model + training logic (OOP)
├── DNN_tensorflow.py       # (optional / comparison)
├── requirements.txt
├── README.md
└── .gitignore
⚙️ Installation
pip install -r requirements.txt

(or manually install: torch, pandas, numpy, scikit-learn, matplotlib, seaborn, streamlit)

▶️ Run the App (Local)
streamlit run app.py

Then open:
👉 http://localhost:8501

🧩 Usage
Upload a CSV file
Select:
Problem type (Classification / Regression)
Target column
(Optional) Adjust parameters:
Network architecture (e.g. [64,128,64])
Learning rate, batch size, epochs, etc.
Click Run Training
View:
Metrics
Training curves
Download predictions as CSV
🔧 Key Parameters
Parameter	Description
units	Hidden layers (e.g. [64,128,64])
activation	relu / tanh / sigmoid / leaky_relu
lr	Learning rate
batch_size	Training batch size
epochs	Number of epochs
dropout_rate	Regularization
batch_norm	Enable batch normalization
patience	Early stopping
📊 Outputs
Classification
Accuracy
Loss & accuracy curves
Regression
MAE / MSE (labeled as r2 in code)
Prediction vs target scatter
Loss curve
⚠️ Notes
units uses eval() → safe for local use, not production
Regression “r2” is actually MSE (kept as-is in code)
All columns except target are used as features
🌐 Deployment
Option 1 — Local
streamlit run app.py
Option 2 — Cloud (Free)

Deploy via **Streamlit Cloud:

Push repo to GitHub
Go to Streamlit Cloud
Select repo + app.py
Deploy