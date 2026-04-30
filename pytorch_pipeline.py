"""
pytorch supervised learning template for classification and regression tasks, 
with early stopping, learning rate scheduling, and comprehensive evaluation metrics.
The code is organized into modular functions for data loading, model definition, 
training, evaluation, and prediction. It supports both classification and regression modes, 
allowing for flexible use across different datasets and tasks.
"""
# system libraries
from csv import writer
import os
import random
import numpy as np
import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt            
# preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
# model selection
from sklearn.model_selection import train_test_split
# Evaluation metrics_for regression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
# Evaluation metrics_for classification
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
# Deep neural network
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter


class SequentialNet(nn.Module):
    def __init__(self, input_dim, hidden_units, output_dim,
                activation="relu", dropout_rate=0.0, batch_norm=False,
                random_seed=42):
        super().__init__()

        # Ensure reproducibility 
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        random.seed(random_seed)
        os.environ['PYTHONHASHSEED'] = str(random_seed)

        act_map = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "leaky_relu": nn.LeakyReLU()
            }
        act = act_map[activation]

        layers = []
        in_dim = input_dim

        for u in hidden_units:
            layers.append(nn.Linear(in_dim, u))
            if batch_norm:
                layers.append(nn.BatchNorm1d(u))
            layers.append(act)
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            in_dim = u

        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class DeepLearningPipeline:
    
    def __init__(self, mode, device=None):
        self.mode = mode
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # model.to(self.device)

    # LOSS & METRICS
    def get_loss_and_metrics(self):
        if self.mode == "Classification":
            loss_fn = nn.CrossEntropyLoss()

            def metrics_fn(logits, y):
                preds = logits.argmax(dim=1)
                acc = (preds == y).float().mean().item()
                return {"accuracy": acc}

        elif self.mode == "Regression":
            loss_fn = nn.MSELoss()

            def metrics_fn(preds, y):
                mae = mean_absolute_error(
                    y.cpu().numpy(), preds.cpu().numpy()
                )
                r2 = r2_score(
                    y.cpu().numpy(), preds.cpu().numpy()
                )
                return {"mae": mae, "r2": r2}

        return loss_fn, metrics_fn


    class EarlyStopping:
        def __init__(self, patience=10, minimize=True, save_path="best.pth"):
            self.patience = patience
            self.minimize = minimize
            self.save_path = save_path
            self.best_loss = np.inf if minimize else -np.inf
            self.counter = 0
            self.should_stop = False

        def step(self, val_loss, model):
            improved = (val_loss < self.best_loss) if self.minimize else (val_loss > self.best_loss)
            if improved:
                self.best_loss = val_loss
                self.counter = 0
                torch.save(model.state_dict(), self.save_path)
                # print(f"Model improved. Saved to {self.save_path}")
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.should_stop = True
                    print("Early stopping triggered!")

    # TRAIN
    def train_model(
        self,
        model,
        train_loader,
        val_loader,
        num_epochs=100,
        optimizer_name='adam',
        lr=1e-4,
        weight_decay=1e-4,
        patience=10,
        save_path="best_model.pth",
        lr_factor=0.5,
        lr_patience=5,

    ):
        model.to(self.device)

        loss_fn, metrics_fn = self.get_loss_and_metrics()

        if optimizer_name == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        elif optimizer_name == 'rmsprop':
            optimizer = optim.RMSprop(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=lr_factor, patience=lr_patience)
        early_stopper = self.EarlyStopping(patience=patience, save_path=save_path)

        train_loss_hist = []
        val_loss_hist = []
        train_acc_hist = []
        val_acc_hist = []

        writer = SummaryWriter(log_dir="./logs")

        for epoch in range(num_epochs):
            # Train
            model.train()
            train_losses = []
            train_acc = []

            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)

                optimizer.zero_grad()
                preds = model(xb)
                loss = loss_fn(preds.squeeze(), yb)
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

                if self.mode == 'Classification':
                    preds = torch.argmax(preds, dim=1)
                    acc = (preds == yb).float().mean()
                    train_acc.append(float(acc.item()))

            train_loss = np.mean(train_losses)
            train_loss_hist.append(train_loss)

            if self.mode == "Classification":
                train_acc_hist.append(np.mean(train_acc))

            # Validation 
            model.eval()
            val_losses = []
            val_acc = []

            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    preds = model(xb)
                    loss = loss_fn(preds.squeeze(), yb)

                    val_losses.append(loss.item())

                    if self.mode == 'Classification':
                        preds = torch.argmax(preds, dim=1)
                        acc = (preds == yb).float().mean()
                        val_acc.append(float(acc.item()))

            val_loss = np.mean(val_losses)
            val_loss_hist.append(val_loss)

            if self.mode == "Classification":
                val_acc_hist.append(np.mean(val_acc))

            scheduler.step(val_loss)
            early_stopper.step(val_loss, model)

            if (epoch+1) % 10 == 0:
                print(
                    f"Epoch:      {epoch+1:03d} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss:   {val_loss:.4f}"
                )

            # writer.add_scalar("Loss/Train", train_loss, epoch)
            # writer.add_scalar("Loss/Validation", val_loss, epoch)
            # writer.add_scalar("LR", optimizer.param_groups[0]['lr'], epoch)

            # if self.mode == "Classification":
            #     writer.add_scalar("Accuracy/Train", np.mean(train_acc), epoch)
            #     writer.add_scalar("Accuracy/Validation", np.mean(val_acc), epoch)
            
            if early_stopper.should_stop:
                break

        # writer.close()

        model.load_state_dict(torch.load(save_path))

        history = {
            "train_loss": train_loss_hist,
            "val_loss": val_loss_hist,
            "train_acc": train_acc_hist if self.mode == "Classification" else None,
            "val_acc": val_acc_hist if self.mode == "Classification" else None,
        }

        return model, history

    def evaluate_model(self, model, dataloader):

        model.to(self.device)

        loss_fn, metrics_fn = self.get_loss_and_metrics()

        model.eval()
        losses = []
        metrics_all = []

        with torch.no_grad():
            for xb, yb in dataloader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                preds = model(xb)

                loss = loss_fn(preds.squeeze(), yb)
                losses.append(loss.item())

                metrics = metrics_fn(preds.squeeze(), yb)
                metrics_all.append(metrics)

        mean_loss = np.mean(losses)

        if self.mode == "Classification":
            acc = np.mean([m["accuracy"] for m in metrics_all])
            return {"loss": mean_loss, "accuracy": acc}

        else:
            mae = np.mean([m["mae"] for m in metrics_all])
            r2 = np.mean([m["r2"] for m in metrics_all])
            return {"loss": mean_loss, "mae": mae, "r2": r2}

    # PREDICTIONS
    def collect_predictions(self, model, dataloader, target_encoder=None):

        model.to(self.device)

        model.eval()

        preds = []
        targets = []

        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                out = model(X)

                if self.mode == "Classification":
                    out = torch.argmax(out, dim=1)

                preds.append(out.cpu().numpy())
                targets.append(y.cpu().numpy())

        preds = np.concatenate(preds)
        targets = np.concatenate(targets)

        if self.mode == "Classification" and target_encoder is not None:
            preds = target_encoder.inverse_transform(preds)
            targets = target_encoder.inverse_transform(targets)

        return preds, targets

    @staticmethod
    def save_predictions_to_csv(df, target_col, preds, output_path):

        df_out = df.copy()
        df_out = df_out.iloc[:len(preds)].copy()
        df_out["predicted_" + target_col] = preds
        df_out.to_csv(output_path, index=False)

        print(f"Saved predictions to {output_path}")

    @staticmethod
    def evaluate_classification_pt(
        pred_labels,
        true_labels,
        target_encoder
    ):
        
        classes = target_encoder.classes_
        # pred_labels = classes[y_pred]
        # true_labels = classes[y_true]

        acc = accuracy_score(true_labels, pred_labels)
        print(f"Samples: {len(pred_labels)}  Accuracy: {acc*100:.2f}%")
        print(classification_report(true_labels, pred_labels))

        cm = confusion_matrix(true_labels, pred_labels)

        plt.figure()
        sb.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=classes,
            yticklabels=classes
        )
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()

        return acc, cm

    @staticmethod
    def evaluate_regression_pt(
        y_pred,
        y_true
    ):
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)

        print(f"r²: {r2:.4f}")
        print(f"mae: {mae:.4f}")

        plt.figure()
        plt.scatter(y_pred, y_true, alpha=0.7)
        plt.xlabel("Predictions")
        plt.ylabel("Targets")
        plt.title("Predictions vs Targets")
        plt.grid(True)
        plt.show()

        return r2, mae

    @staticmethod
    def plot_training_curves_pt(
        history
    ):
        
        train_loss_hist = history["train_loss"]
        val_loss_hist  = history["val_loss"]
        train_acc_hist  = history["train_acc"]
        val_acc_hist   = history["val_acc"]

        plt.figure()
        plt.plot(train_loss_hist, label="train")
        plt.plot(val_loss_hist, label="validation")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.show()

        if train_acc_hist is not None and val_acc_hist is not None:
            plt.figure()
            plt.plot(train_acc_hist, label="train")
            plt.plot(val_acc_hist, label="validation")
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.grid(True)
            plt.show()


    def predict_from_checkpoint(
        self,
        model_name,
        input_dim,
        hidden_units,
        output_dim,
        X,
        activation,
        target_encoder,
        dropout_rate,
        batch_norm,
        random_seed
    ):


        model = SequentialNet(
            input_dim=input_dim,
            hidden_units=hidden_units,
            output_dim=output_dim,
            activation=activation,
            dropout_rate=dropout_rate,
            batch_norm=batch_norm,
            random_seed=random_seed
        )

        model.load_state_dict(torch.load(model_name, map_location=self.device))
        model.to(self.device)
        model.eval()

        X = torch.tensor(X, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            preds = model(X)

            if self.mode == "Classification":
                preds = preds.argmax(dim=1).cpu().numpy()
                if target_encoder is not None:
                    preds = target_encoder.inverse_transform(preds)
            else:
                preds = preds.cpu().numpy()

        return preds

    def load_and_preprocess_data(
        self,
        filepath,
        target_col,
        test_size=0.2,
        random_seed=42,
        batch_size=32
    ):
        df = pd.read_csv(filepath)

        X = df.drop(columns=[target_col]).copy()
        y = df[target_col].copy()

        # X = df.iloc[:, :-1].copy()
        # y = df.iloc[:,-1].copy()

        # Handle non-numeric features
        for col in X.columns:
            if not np.issubdtype(X[col].dtype, np.number):
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])

        #Target encoding (classification only) 
        target_encoder = None
        if self.mode == "Classification":
            target_encoder = LabelEncoder()
            y = target_encoder.fit_transform(y)
            y = torch.tensor(y, dtype=torch.long)
        else:
            y = torch.tensor(y.values, dtype=torch.float32)

        # Train/test split
        Xtrain, Xtest, Ytrain, Ytest = train_test_split(
            X, y, test_size=test_size, random_state=random_seed, shuffle=True
        )

        # Scaling 
        scaler = StandardScaler()
        Xtrain = scaler.fit_transform(Xtrain)
        Xtest = scaler.transform(Xtest)
        X = scaler.transform(X)

        # Convert to tensors
        Xtrain = torch.tensor(Xtrain, dtype=torch.float32)
        Xtest = torch.tensor(Xtest, dtype=torch.float32)

        if not isinstance(Ytrain, torch.Tensor):
            Ytrain = torch.tensor(Ytrain)
            Ytest = torch.tensor(Ytest)


        train_dataloader = DataLoader(TensorDataset(Xtrain, Ytrain), batch_size=batch_size, shuffle=True)
        test_dataloader  = DataLoader(TensorDataset(Xtest, Ytest), batch_size=batch_size, shuffle=False)

        input_dim = Xtrain.shape[1]
        output_dim = len(np.unique(Ytrain.numpy())) if self.mode=="Classification" else 1

        return train_dataloader, test_dataloader, X, Xtest.numpy(), input_dim, output_dim, scaler, target_encoder, df



if __name__ == '__main__':


    # mode = "Classification"
    mode='Regression'

    # Parameters
    random_seed=42
    activation='relu'
    optimizer='adam'#'sgd'#'rmsprop'
    units=[64,128,64]
    lr=0.0001 
    dropout_rate=0#0.1
    batch_norm=False#True
    # l2_reg=None#1e-4
    # callback=False#True
    patience=10
    lr_factor=0.5
    lr_patience=5
    model_name='best_model.pth'
    # log_dir='./logs'
    epochs=30
    test_size=0.2
    batch_size=8
    # verbose=1
    weight_decay=1e-4#l2_reg


    if mode=='Regression':
        CSV_PATH=r"E:/AB/ai_ml_apps_lab_github_2026/3Pytorch/housing.csv"
        target_col = "price"

    elif mode=='Classification':
        CSV_PATH=r"E:/AB/ai_ml_apps_lab_github_2026/3Pytorch/iris.csv"
        feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"] 
        target_col = "species"

    pipe = DeepLearningPipeline(mode=mode)

    train_dataloader, test_dataloader, X, Xtest, input_dim, output_dim, scaler, target_encoder, df = \
        pipe.load_and_preprocess_data(
            CSV_PATH,
            target_col,
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

    model, history = pipe.train_model(
        model,
        train_dataloader,
        test_dataloader,
        num_epochs=epochs,
        optimizer_name=optimizer,
        lr=lr,
        weight_decay=weight_decay,
        patience=patience,
        save_path=model_name,
        lr_factor=lr_factor,
        lr_patience=lr_patience,
    )

    results = pipe.evaluate_model(model, test_dataloader)

    # Collect predictions
    y_pred, y_true = pipe.collect_predictions(model, test_dataloader, target_encoder)

    # Evaluation
    if mode == "Classification":
        acc, cm = pipe.evaluate_classification_pt(
            y_pred,
            y_true,
            target_encoder
        )
    else:
        r2, mae = pipe.evaluate_regression_pt(
            y_pred,
            y_true
        )

    # Training curves 
    pipe.plot_training_curves_pt(
        history
    )

    # Prediction only (no training code involved)
    preds = pipe.predict_from_checkpoint(
        model_name=model_name,
        input_dim=input_dim,
        hidden_units=units,
        output_dim=output_dim,
        X=Xtest,
        activation=activation,
        target_encoder=target_encoder,
        dropout_rate=dropout_rate,
        batch_norm=batch_norm,
        random_seed=random_seed
    )


# public web app 

