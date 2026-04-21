"""
DNN with tensorflow
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
# model selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
# Evaluation metrics_for regression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
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
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Concatenate, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard



# Heteroscedastic loss function (Kendal and Gal, 2017)
def hs_loss(y_true, y_pred):
    N = y_true.shape[0]
    se = K.pow((y_true[:,0]-y_pred[:,0]),2)
    inv_std = K.exp(-y_pred[:,1])
    mse = K.mean(inv_std*se)
    reg = K.mean(y_pred[:,1])
    loss = 0.5 * (mse + reg)
    return loss


def get_loss_and_metrics(mode, model_type):
    if mode == "Regression":
        metrics = ["mae"]
        if model_type == "Sequential":
            loss = "mse"
        else:
            loss = hs_loss

    elif mode == "Classification":
        metrics = ["accuracy"]
        loss = "sparse_categorical_crossentropy"
        # 'sparse_categorical_crossentropy', categorical_crossentropy, 'binary_crossentropy'

    return loss, metrics

def build_dnn(
    mode,
    model_type,
    input_dim,
    output_dim,
    units,
    activation="relu",
    dropout_rate=0.0,
    batch_norm=False,
    l2_reg=None
):
    
    # Notes:
    # Architecture choice for hidden layer units (funnel, zigzag, constant)
    # Validation-driven tuning to avoid over/underfitting (high variance/high bias)
    # Dependence on dataset size
    # Use of regularization (dropout, L2 weight decay, batch normalization, reduce LR, early stopping) to control overfitting

    # Guidance
    # For each epoch →
    # For each mini-batch →
    #  Forward pass through layers →
    #  Loss computation →
    #  Backward pass through layers →
    #  Update Wᶦ and bᶦ for each layer

    # Ensure reproducibility 
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)

    # For Uncertainty Quantification
    if model_type == "Model":  # heteroscedastic
        inputs = Input(shape=(input_dim,))
        x = inputs

        for i, u in enumerate(units):
            x = Dense(
                u,
                activation=activation,
                kernel_regularizer=l2(l2_reg) if l2_reg else None
            )(x)

        mean = Dense(1, name="mean")(x)
        log_var = Dense(1, name="log_var")(x)
        outputs = Concatenate()([mean, log_var])

        return Model(inputs, outputs)

    # ---- Sequential ----
    model = Sequential()

    for i, u in enumerate(units):
        if i == 0:
            model.add(Dense(
                u,
                activation=activation,
                input_shape=(input_dim,),
                kernel_regularizer=l2(l2_reg) if l2_reg else None
            ))
        else:
            model.add(Dense(
                u,
                activation=activation,
                kernel_regularizer=l2(l2_reg) if l2_reg else None
            ))

        if batch_norm:
            model.add(BatchNormalization())
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))

    out_act = "softmax" if mode == "Classification" else None
    model.add(Dense(output_dim, activation=out_act))

    return model


def compile_model(model, mode, model_type, optimizer, lr):
    loss, metrics = get_loss_and_metrics(mode, model_type)

    if optimizer == "adam":
        opt = Adam(lr)
    elif optimizer == "sgd":
        opt = SGD(lr, momentum=0.9)
    elif optimizer == "rmsprop":
        opt = RMSprop(lr)
    else:
        opt = optimizer

    model.compile(optimizer=opt, loss=loss, metrics=metrics)
    return model


def train_model(
    model,
    Xtrain,
    Ytrain,
    Xval,
    Yval,
    epochs,
    batch_size,
    patience,
    lr_factor,
    save_path,
    log_dir,
    verbose=1
):
    callbacks = [
        EarlyStopping(patience=patience, restore_best_weights=True),
        ModelCheckpoint(save_path, save_best_only=True),
        ReduceLROnPlateau(factor=lr_factor, patience=patience),
        TensorBoard(log_dir=log_dir)
    ]

    history = model.fit(
        Xtrain, Ytrain,
        validation_data=(Xval, Yval),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=verbose
    )
    return history


def evaluate_model(model, Xtest, Ytest, mode):
    results = model.evaluate(Xtest, Ytest, verbose=0)

    if mode == "Regression":
        return {"loss": results[0], "mae": results[1]}
    else:
        return {"loss": results[0], "accuracy": results[1]}


def evaluate_regression(
    model,
    Xtrain, Ytrain,
    Xtest, Ytest,
    model_type="Sequential"
):
    # ---- Predictions ----
    ypred_train = model.predict(Xtrain)
    ypred_test = model.predict(Xtest)

    # ---- Heteroscedastic case ----
    if model_type == "Model":
        ypred_train, sigma_train = ypred_train[:, 0], ypred_train[:, 1]
        ypred_test, sigma_test = ypred_test[:, 0], ypred_test[:, 1]

    # ---- Metrics ----
    r2_train = r2_score(Ytrain, ypred_train) * 100
    rmse_train = np.sqrt(mean_squared_error(Ytrain, ypred_train))

    r2_test = r2_score(Ytest, ypred_test) * 100
    rmse_test = np.sqrt(mean_squared_error(Ytest, ypred_test))

    print(f"Train R² / RMSE: {r2_train:.2f}, {rmse_train:.4f}")
    print(f"Test  R² / RMSE: {r2_test:.2f}, {rmse_test:.4f}")

    return {
        "train": {"r2": r2_train, "rmse": rmse_train},
        "test": {"r2": r2_test, "rmse": rmse_test}
    }

def plot_training_history(history):
    plt.figure()
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training History")
    plt.legend()
    plt.grid(True)
    plt.show()

def evaluate_classification(
    model,
    X,
    y_true,
    class_labels=None
):
    ypred_prob = model.predict(X)
    ypred = np.argmax(ypred_prob, axis=1)

    acc = accuracy_score(y_true, ypred)
    print(f"Accuracy: {acc*100:.2f}%")
    print(classification_report(y_true, ypred))

    cm = confusion_matrix(y_true, ypred)

    plt.figure()
    sb.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_labels,
        yticklabels=class_labels
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    return acc, cm


def predict_from_saved_model(
    model_path,
    X,
    mode,
    model_type=None
):
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={"hs_loss": hs_loss},
        compile=False
    )
    preds = model.predict(X)

    if mode == "Classification":
        preds = np.argmax(preds, axis=1)

    return preds


if __name__ == '__main__':

    mode='Regression'
    # mode='Classification'

    model_type = 'Sequential'
    # model_type = 'Model' # For Regression

    # Parameters
    random_seed=42
    test_size=0.2
    activation='relu'
    optimizer='adam'
    units=[64,128,64]
    lr=0.0001 
    dropout_rate=0#0.1
    batch_norm=False#True
    l2_reg=None#1e-4
    # metrics=None
    # loss=None
    callback=False#True
    patience=10
    lr_factor=0.5
    h5='best_model.keras'
    log_dir='./logs'
    epochs=30
    batch_size=8
    verbose=1

    if mode=='Regression':
        filepath=r"E:/AB/ai_ml_apps_lab_github_2026/3Pytorch/housing.csv"
        dataset=pd.read_csv(filepath)
        # dataset=dataframe.values
        X=dataset.iloc[:,:13]
        Y=dataset.iloc[:,13]
        encoded_Y=Y

    elif mode=='Classification':
        filepath=r"E:/AB/ai_ml_apps_lab_github_2026/3Pytorch/iris.csv"
        dataset=pd.read_csv(filepath)
        X=dataset.iloc[:,:4].astype(float)
        Y=dataset.iloc[:,4]

        encoder = LabelEncoder()
        encoder.fit(Y)
        encoded_Y = encoder.transform(Y)
        # Y = to_categorical(encoded_Y).astype(float)

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,encoded_Y,test_size=test_size,random_state=random_seed)
    scale=StandardScaler()
    Xtrain=scale.fit_transform(Xtrain).astype(float)
    Xtest=scale.transform(Xtest).astype(float)
    Ytrain=Ytrain.astype(int)
    Ytest=Ytest.astype(int)

    
    input_dim = int(X.shape[1])
    if mode=='Regression':
        if model_type == 'Model':
            output_dim = 2
        else:
            output_dim = 1

    elif mode=='Classification':
        output_dim = len(np.unique(encoded_Y)) #int(Y.shape[1]) #len(np.unique(encoded_Y))
    

    model = build_dnn(
        mode=mode,
        model_type=model_type,
        input_dim=input_dim,
        output_dim=output_dim,
        units=units,
        activation=activation,
        dropout_rate=dropout_rate,
        batch_norm=batch_norm,
        l2_reg=l2_reg
    )

    model = compile_model(
        model,
        mode=mode,
        model_type=model_type,
        optimizer=optimizer,
        lr=lr
    )

    history = train_model(
        model,
        Xtrain, Ytrain,
        Xtest, Ytest,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        lr_factor=lr_factor,
        save_path=h5,
        log_dir=log_dir
    )

    results = evaluate_model(model, Xtest, Ytest, mode)
    print(results)

    if mode == "Regression":
        metrics = evaluate_regression(
            model,
            Xtrain, Ytrain,
            Xtest, Ytest,
            model_type=model_type
        )
        plot_training_history(history)

    elif mode == "Classification":
        acc, cm = evaluate_classification(
            model,
            Xtest,
            Ytest,
            class_labels=np.unique(encoded_Y)
        )

    # ---- Inference only ----
    preds = predict_from_saved_model(h5, Xtest, mode, model_type=model_type)

