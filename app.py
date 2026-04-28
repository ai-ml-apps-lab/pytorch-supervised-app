import streamlit as st
import pandas as pd
import numpy as np
import torch

from pipeline import SequentialNet, DeepLearningPipeline


st.set_page_config(layout="wide")
st.title("🧠 PyTorch Deep Learning UI")

# =========================
# SIDEBAR CONFIG
# =========================
st.sidebar.header("⚙️ Configuration")

# Upload CSV
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.warning("Please upload a CSV file")
    st.stop()

st.write("### 📄 Dataset Preview")
st.dataframe(df.head())

columns = df.columns.tolist()

# Mode
mode = st.sidebar.selectbox("Mode", ["Classification", "Regression"])

# Target column
TARGET_COL = st.sidebar.selectbox("Target Column", columns)

# =========================
# PARAMETERS
# =========================
st.sidebar.subheader("Model Parameters")

random_seed = st.sidebar.number_input("Random Seed", value=42)

activation = st.sidebar.selectbox(
    "Activation", ["relu", "tanh", "sigmoid", "leaky_relu"]
)

optimizer_name = st.sidebar.selectbox("Optimizer", ["adam"])

units_str = st.sidebar.text_input("Hidden Units", "[64,128,64]")
units = eval(units_str)

lr = st.sidebar.number_input("Learning Rate", value=0.0001, format="%.5f")

dropout_rate = st.sidebar.number_input("Dropout", value=0.0)

batch_norm = st.sidebar.selectbox("Batch Norm", [False, True])

l2_reg = st.sidebar.text_input("L2 Reg (None or value)", "None")
l2_reg = None if l2_reg == "None" else float(l2_reg)

callback = st.sidebar.selectbox("Callback (EarlyStopping)", [False, True])

patience = st.sidebar.number_input("Patience", value=10)

lr_factor = st.sidebar.number_input("LR Factor", value=0.5)

lr_patience = st.sidebar.number_input("LR Patience", value=5)

model_name = st.sidebar.text_input("Model Name", "best_model.pth")

log_dir = st.sidebar.text_input("Log Dir", "./logs")

epochs = st.sidebar.number_input("Epochs", value=30)

test_size = st.sidebar.number_input("Test Size", value=0.2)

random_state = st.sidebar.number_input("Random State", value=42)

batch_size = st.sidebar.number_input("Batch Size", value=8)

verbose = st.sidebar.number_input("Verbose", value=1)

weight_decay = st.sidebar.number_input("Weight Decay", value=1e-4, format="%.6f")


# =========================
# RUN BUTTON
# =========================
if st.button("🚀 Run Training"):

    pipe = DeepLearningPipeline(mode=mode)

    # Save uploaded file temporarily
    temp_path = "temp.csv"
    df.to_csv(temp_path, index=False)

    train_loader, test_loader, Xtest, input_dim, output_dim, scaler, target_encoder, df_full = \
        pipe.load_and_preprocess_data(
            temp_path,
            feature_cols=None,
            target_col=TARGET_COL,
            test_size=test_size,
            random_state=random_state,
            batch_size=batch_size
        )

    model = SequentialNet(
        input_dim=input_dim,
        hidden_units=units,
        output_dim=output_dim,
        activation=activation,
        dropout_rate=dropout_rate,
        batch_norm=batch_norm,
        random_seed=random_seed
    )

    st.write("### 🏋️ Training...")

    model, history = pipe.train_model(
        model,
        train_loader,
        test_loader,
        num_epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        patience=patience,
        save_path=model_name,
        lr_factor=lr_factor,
        lr_patience=lr_patience,
    )

    st.success("Training Completed!")

    # =========================
    # EVALUATION
    # =========================
    results = pipe.evaluate_model(model, test_loader)

    st.write("### 📊 Results")
    st.json(results)

    # =========================
    # PREDICTIONS
    # =========================
    y_pred, y_true = pipe.collect_predictions(
        model, test_loader, target_encoder
    )

    # =========================
    # PLOTS
    # =========================
    st.subheader("📉 Training Curves")

    st.line_chart({
        "train_loss": history["train_loss"],
        "val_loss": history["val_loss"]
    })

    if mode == "Classification" and history["train_acc"] is not None:
        st.line_chart({
            "train_acc": history["train_acc"],
            "val_acc": history["val_acc"]
        })

    # =========================
    # TASK-SPECIFIC VISUALS
    # =========================
    if mode == "Classification":
        st.subheader("📊 Classification Report")

        acc = (y_pred == y_true).mean()
        st.write(f"Accuracy: {acc:.4f}")

    else:
        st.subheader("📊 Regression Metrics")

        r2 = ((y_true - y_pred)**2).mean()
        mae = np.abs(y_true - y_pred).mean()

        st.write(f"MAE: {mae:.4f}")
        st.write(f"MSE: {r2:.4f}")

        st.scatter_chart(pd.DataFrame({
            "Predictions": y_pred.flatten(),
            "Targets": y_true.flatten()
        }))

    # =========================
    # SAVE CSV
    # =========================
    st.subheader("💾 Download Predictions")

    df_out = df.copy()
    df_out = df_out.iloc[:len(y_pred)].copy()
    df_out["predicted_" + TARGET_COL] = y_pred

    csv = df_out.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="Download CSV with Predictions",
        data=csv,
        file_name="predictions.csv",
        mime="text/csv",
    )