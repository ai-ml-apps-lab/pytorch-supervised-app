
"""
This is the main Streamlit app for the PyTorch Deep Learning Pipeline.
"""
#Libraries
import streamlit as st
import pandas as pd
import numpy as np
import time

from pytorch_pipeline import SequentialNet, DeepLearningPipeline

st.set_page_config(layout="wide")
st.title("PyTorch Deep Learning App")

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

# Configuration
st.sidebar.subheader("Model Configuration")

col1, col2, col3 = st.sidebar.columns(3)

with col1:

    # Mode
    mode = st.selectbox("Mode", ["Classification", "Regression"])

    # Target column
    target_col = st.selectbox("Target Column", columns)

    activation = st.selectbox(
        "Activation", ["relu", "tanh", "sigmoid", "leaky_relu"]
    )

    optimizer = st.selectbox("Optimizer", ["adam", "sgd", "rmsprop"])

    test_size = st.number_input("Test Size", value=0.2)

    batch_size = st.number_input("Batch Size", value=16)

with col2:

    epochs = st.number_input("Epochs", value=100)

    units_str = st.text_input("Hidden Units", "[64,128,64]")
    units = eval(units_str)

    lr = st.number_input("Learning Rate", value=0.0001, format="%.5f")

    random_seed = st.number_input("Random Seed", value=42)

    batch_norm = st.selectbox("Batch Norm", [False, True])


with col3:

    dropout_rate = st.number_input("Dropout", value=0.0)

    patience = st.number_input("Patience", value=10)

    lr_factor = st.number_input("LR Factor", value=0.5)

    lr_patience = st.number_input("LR Patience", value=10)

    weight_decay = st.number_input("Weight Decay", value=1e-4, format="%.6f")


# RUN BUTTON
if st.button("🚀 Run Training"):

    pipe = DeepLearningPipeline(mode=mode)

    # Save uploaded file temporarily
    temp_path = "temp.csv"
    df.to_csv(temp_path, index=False)

    train_loader, test_loader, X, Xtest, input_dim, output_dim, scaler, target_encoder, df_full = \
        pipe.load_and_preprocess_data(
            temp_path,
            target_col=target_col,
            test_size=test_size,
            random_seed=random_seed,
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

    save_path = f"model_{int(time.time())}.pth"

    model, history = pipe.train_model(
        model,
        train_loader,
        test_loader,
        num_epochs=epochs,
        optimizer_name=optimizer,
        lr=lr,
        weight_decay=weight_decay,
        patience=patience,
        save_path=save_path,
        lr_factor=lr_factor,
        lr_patience=lr_patience,
    )

    st.success("Training Completed!")

    # EVALUATION
    results = pipe.evaluate_model(model, test_loader)

    st.write("### 📊 Results")
    st.json(results)

    # PREDICTIONS
    y_pred, y_true = pipe.collect_predictions(
        model, test_loader, target_encoder
    )

    # PLOTS
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

    # TASK-SPECIFIC VISUALS
    if mode == "Classification":
        st.subheader("📊 Classification Report")

        acc, cm = pipe.evaluate_classification_pt(
            y_pred,
            y_true,
            target_encoder
        )
        st.write(f"Accuracy: {acc:.4f}")

    else:
        st.subheader("📊 Regression Metrics")

        r2, mae = pipe.evaluate_regression_pt(
            y_pred,
            y_true
        )

        st.write(f"MAE: {mae:.4f}")
        st.write(f"R2: {r2:.4f}")

    # SAVE CSV
    st.subheader("💾 Download Predictions")

    preds = pipe.predict_from_checkpoint(
        save_path,
        input_dim,
        units,
        output_dim,
        X,
        activation,
        target_encoder,
        dropout_rate,
        batch_norm,
        random_seed
    )

    df_out = df.copy()
    df_out["predicted_" + target_col] = preds

    csv = df_out.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="Download CSV with Predictions",
        data=csv,
        file_name="predictions.csv",
        mime="text/csv",
    )
