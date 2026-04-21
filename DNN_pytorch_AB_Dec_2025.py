"""
DNN with pytorch
AB
Dec, 2025
"""
# system libraries
import os
import sys
import copy
import pickle
import math
import random
import numpy as np
import pandas as pd
import lasio
import seaborn as sb
from matplotlib import pyplot as plt            
# preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
# model selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
# Evaluation metrics_for regression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
# Evaluation metrics_for classification
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
# ML algorithms _ regression
from sklearn import linear_model 
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
# ML algorithms _ classification
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
# from scikeras.wrappers import KerasRegressor
# Deep neural network
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader
# from tensorflow.keras.utils import to_categorical

# from torchvision import datasets
# from torchvision.transforms import ToTensor
# import tqdm 


class SequentialNet(nn.Module):
    def __init__(self, input_dim, hidden_units, output_dim,
                activation="relu", dropout_rate=0.0, batch_norm=False):
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
            # "softmax": nn.Softmax(dim=1)
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

def get_loss_and_metrics(mode):
    if mode == "Classification":
        loss_fn = nn.CrossEntropyLoss()

        def metrics_fn(logits, y):
            preds = logits.argmax(dim=1)
            acc = (preds == y).float().mean().item()
            return {"accuracy": acc}

    elif mode == "Regression":
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
            print(f"Model improved. Saved to {self.save_path}")
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                print("Early stopping triggered!")


def train_model(
    model,
    train_loader,
    val_loader,
    mode,
    num_epochs=100,
    lr=1e-4,
    weight_decay=1e-4,
    patience=10,
    save_path="best_model.pth"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    loss_fn, metrics_fn = get_loss_and_metrics(mode)

    optimizer = optim.Adam(model.parameters(), lr=lr)#, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    early_stopper = EarlyStopping(patience=patience, save_path=save_path)

    history = {"train_loss": [], "val_loss": []}

    train_loss_hist = []
    val_loss_hist = []
    train_acc_hist = []
    val_acc_hist = []

    for epoch in range(num_epochs):
        # ---- Train ----
        model.train()
        train_losses = []
        test_losses = []
        train_acc = []
        test_acc = []

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb.unsqueeze(1))
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

            if mode == 'Classification':
                preds = torch.argmax(preds, dim=1)
                acc = (preds == yb).float().mean()
                train_acc.append(float(acc.item()))

        train_loss = np.mean(train_losses)
        train_loss_hist.append(train_loss)

        if mode == "Classification":
            train_acc_hist.append(train_acc)

        # ---- Validation ----
        model.eval()

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                loss = loss_fn(preds, yb.unsqueeze(1)).item()
                test_losses.append(loss)

                if mode == 'Classification':
                    acc = float((torch.argmax(preds, dim=1) == yb).float().mean().item())
                    test_acc.append(acc)

        val_loss = np.mean(test_losses)
        val_loss_hist.append(val_loss)

        if mode == "Classification":
            val_acc_hist.append(test_acc)

        scheduler.step(val_loss)
        early_stopper.step(val_loss, model)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(
            f"Epoch {epoch+1:03d} | "
            f"Train: {train_loss:.4f} | "
            f"Val: {val_loss:.4f}"
        )

        if early_stopper.should_stop:
            break

    model.load_state_dict(torch.load(save_path))

    history = {
        "train_loss": train_loss_hist,
        "val_loss": val_loss_hist,
        "train_acc": train_acc_hist if mode == "Classification" else None,
        "val_acc": val_acc_hist if mode == "Classification" else None,
    }

    return model, history

def evaluate_model(model, dataloader, mode):
    device = next(model.parameters()).device
    loss_fn, metrics_fn = get_loss_and_metrics(mode)

    model.eval()
    losses = []
    metrics_all = []

    with torch.no_grad():
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)

            losses.append(loss_fn(preds, yb.unsqueeze(1)).item())
            metrics_all.append(metrics_fn(preds, yb))

    mean_loss = np.mean(losses)

    if mode == "Classification":
        acc = np.mean([m["accuracy"] for m in metrics_all])
        return {"loss": mean_loss, "accuracy": acc}

    else:
        mae = np.mean([m["mae"] for m in metrics_all])
        r2 = np.mean([m["r2"] for m in metrics_all])
        return {"loss": mean_loss, "mae": mae, "r2": r2}

def collect_predictions_pt(model, dataloader, mode):
    device = next(model.parameters()).device
    model.eval()

    preds = []
    targets = []

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            out = model(X)

            if mode == "Classification":
                out = torch.argmax(out, dim=1)

            preds.append(out.cpu().numpy())
            targets.append(y.cpu().numpy())

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    return preds, targets

def evaluate_classification_pt(
    y_pred,
    y_true,
    classes
):
    pred_labels = classes[y_pred]
    true_labels = classes[y_true]

    acc = accuracy_score(true_labels, pred_labels)
    print(f"Samples: {len(y_pred)}  Accuracy: {acc*100:.2f}%")
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

def evaluate_regression_pt(
    y_pred,
    y_true
):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    print(f"R²: {r2:.4f}")
    print(f"MAE: {mae:.4f}")

    plt.figure()
    plt.scatter(y_pred, y_true, alpha=0.7)
    plt.xlabel("Predictions")
    plt.ylabel("Targets")
    plt.title("Predictions vs Targets")
    plt.grid(True)
    plt.show()

    return r2, mae

def plot_training_curves_pt(
    history
):
    
    train_loss_hist = history["train_loss"]
    test_loss_hist  = history["val_loss"]
    train_acc_hist  = history["train_acc"]
    test_acc_hist   = history["val_acc"]

    plt.figure()
    plt.plot(train_loss_hist, label="train")
    plt.plot(test_loss_hist, label="test")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    if train_acc_hist is not None and test_acc_hist is not None:
        plt.figure()
        plt.plot(train_acc_hist, label="train")
        plt.plot(test_acc_hist, label="test")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        plt.show()


def predict_from_checkpoint(
    checkpoint_path,
    input_dim,
    hidden_units,
    output_dim,
    X,
    mode,
    activation="relu"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SequentialNet(
        input_dim=input_dim,
        hidden_units=hidden_units,
        output_dim=output_dim,
        activation=activation
    )
    model.load_state_dict(torch.load(checkpoint_path))
    model.to(device)
    model.eval()

    X = torch.tensor(X, dtype=torch.float32).to(device)

    with torch.no_grad():
        preds = model(X)

        if mode == "Classification":
            preds = preds.argmax(dim=1).cpu().numpy()
        else:
            preds = preds.cpu().numpy()

    return preds



if __name__ == '__main__':

    mode='Regression'
    # mode='Classification'

    # model = 'Sequential'
    # model = 'Model' # For Regression

    # Parameters
    random_seed=42
    activation='relu'
    optimizer='adam'
    units=[64,128,64]
    # units = [8, 8]    
    lr=0.0001 
    dropout_rate=0#0.1
    batch_norm=False#True
    l2_reg=None#1e-4
    # metrics=None
    # loss=None
    callback=False#True
    patience=10
    lr_factor=0.5
    h5='best_model.h5'
    log_dir='./logs'
    epochs=30
    batch_size=8
    verbose=1
    weight_decay=1e-4

    # device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    device = "cpu"
    print(f"Using {device} device")

    if mode=='Regression':
        # filepath=r"C:\Users\Lenovo\Desktop\DNN_2025\housing.csv"
        filepath=r"E:/AB/ai_ml_apps_lab_github_2026/3Pytorch/housing.csv"
        dataset=pd.read_csv(filepath)
        # dataset=dataframe.values
        X=dataset.iloc[:,:13]
        Y=list(dataset.iloc[:,13])
        # from sklearn.datasets import fetch_california_housing
        # data = fetch_california_housing()
        # X, Y = data.data, data.target
        dtype = torch.float32

        # units.insert(0, X.shape[1])
        # units.append(1)
        print(f"units are:{units}")

        input_dim = X.shape[1]
        output_dim = 1

        # loss_fn = nn.MSELoss()

    elif mode=='Classification':
        # filepath=r"C:\Users\Lenovo\Desktop\DNN_2025\iris.csv"
        filepath=r"E:/AB/ai_ml_apps_lab_github_2026/3Pytorch/iris.csv"
        dataset=pd.read_csv(filepath)
        X=dataset.iloc[:,:4].astype(float)
        Y=list(dataset.iloc[:,4])

        encoder = LabelEncoder()
        encoder.fit(Y)
        encoded_Y = encoder.transform(Y)
        Y = torch.from_numpy(encoded_Y).long()
        dtype = torch.long

        last_unit=len(np.unique(encoded_Y))
        # units.insert(0, X.shape[1])
        # units.append(last_unit)
        print(f"units are:{units}")

        input_dim = X.shape[1]
        output_dim = last_unit
        classes = np.unique(Y.numpy())

        # num_classes = len(np.unique(encoded_Y))
        # Y_onehot = F.one_hot(torch.tensor(encoded_Y, dtype=torch.long), num_classes).float()
        # Y_float = torch.tensor(encoded_Y, dtype=torch.float).unsqueeze(1)  
        # loss_fn = nn.BCEWithLogitsLoss()   
        # ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(Y)
        # ohe.fit(Y)
        # y = ohe.transform(Y)

        # loss_fn = nn.CrossEntropyLoss()


    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,Y,test_size=0.2,random_state=0, shuffle=True)
    scale=StandardScaler()
    Xtrain=scale.fit_transform(Xtrain)
    Xtest=scale.transform(Xtest)

    Xtrain = torch.tensor(Xtrain, dtype=torch.float32) 
    Xtest = torch.tensor(Xtest, dtype=torch.float32) 
    Ytrain = torch.tensor(Ytrain, dtype=dtype)
    Ytest = torch.tensor(Ytest, dtype=dtype)

    # n_epochs = 100
    # batch_size = 64
    # batch_start = torch.arange(0, len(Xtrain), batch_size)

    train_dataset = TensorDataset(Xtrain, Ytrain)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TensorDataset(Xtest, Ytest)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    

    model = SequentialNet(
        input_dim=input_dim,
        hidden_units=units,
        output_dim=output_dim,
        activation="relu"
    )

    model, history = train_model(
        model,
        train_dataloader,
        test_dataloader,
        mode=mode,
        num_epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        patience=patience,
        save_path="best_model.pth"
    )

    results = evaluate_model(model, test_dataloader, mode)
    print(results)

    # ---- Collect predictions ----
    y_pred, y_true = collect_predictions_pt(model, test_dataloader, mode)

    # ---- Evaluation ----
    if mode == "Classification":
        acc, cm = evaluate_classification_pt(
            y_pred,
            y_true,
            classes
        )
    else:
        r2, mae = evaluate_regression_pt(
            y_pred,
            y_true
        )

    # ---- Training curves ----
    plot_training_curves_pt(
        history
    )


    # Prediction only (no training code involved)
    preds = predict_from_checkpoint(
        "best_model.pth",
        input_dim=input_dim,
        hidden_units=units,
        output_dim=output_dim,
        X=Xtest.numpy(),
        mode=mode
    )

