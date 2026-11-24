from model import create_model
from read_data import read_data
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt


def force_numeric_col(df, col_name):
    df[col_name] = pd.to_numeric(df[col_name], errors="raise")

def preprocess(data):
    data = pd.DataFrame(data)
    data = data.replace("", np.nan) # convert empty strings to nan for fillna()
    data = data.replace("yes", float(1.0))
    data = data.replace("no", float(0.0))
    data = data.fillna({"job": "unknown"})
    data = data.fillna({"education": "tertiary"})
    data = data.fillna({"contact": "telephone"})
    data = data.fillna({"poutcome": "other"})
    data = data.drop(columns="duration")
    data = pd.get_dummies(data, columns=["job", "marital", "education", "contact", "poutcome", "month", "day_of_week"], dtype=float)

    force_numeric_col(data, "age")
    force_numeric_col(data, "default")
    force_numeric_col(data, "housing")
    force_numeric_col(data, "loan")
    force_numeric_col(data, "balance")
    force_numeric_col(data, "campaign")
    force_numeric_col(data, "pdays")
    force_numeric_col(data, "previous")
    force_numeric_col(data, "y")
    non_numeric_cols = data.select_dtypes(exclude='number').columns.tolist()
    print("non num: ", non_numeric_cols)

    y = data["y"]
    X = data.drop(columns="y")

    age_scaler = StandardScaler()
    balance_scaler = StandardScaler()
    campaign_scaler = StandardScaler()
    pdays_scaler = MinMaxScaler()
    previous_scaler = MinMaxScaler()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1.0/5.0, random_state=42)
    X_train[["age"]] = age_scaler.fit_transform(X_train[["age"]])
    X_train[["balance"]] = balance_scaler.fit_transform(X_train[["balance"]])
    X_train[["campaign"]] = campaign_scaler.fit_transform(X_train[["campaign"]])
    X_train[["pdays"]] = pdays_scaler.fit_transform(X_train[["pdays"]])
    X_train[["previous"]] = previous_scaler.fit_transform(X_train[["previous"]])
    # X_train[["duration"]] = StandardScaler().fit_transform(X_train[["duration"]])

    # transform validation inputs using training input's mean and stddev
    X_test[["age"]] = age_scaler.transform(X_test[["age"]])
    X_test[["balance"]] = balance_scaler.transform(X_test[["balance"]])
    X_test[["campaign"]] = campaign_scaler.transform(X_test[["campaign"]])
    X_test[["pdays"]] = pdays_scaler.transform(X_test[["pdays"]])
    X_test[["previous"]] = previous_scaler.transform(X_test[["previous"]])
    return X_train, X_test, y_train, y_test, \
            balance_scaler, campaign_scaler, pdays_scaler, previous_scaler, age_scaler

def plot_training_history(history):
    # Plot the training and validation accuracy and loss
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Accuracy Plot
    ax1.plot(history.history['accuracy'], label='Train Accuracy', color='royalblue', marker='o')
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy', color='orange', marker='s')
    ax1.set_title('Training & Validation Accuracy')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.set_xticks(range(len(history.history['accuracy'])))
    ax1.set_xticklabels(range(1, len(history.history['accuracy']) + 1))
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)

    # 2. Loss Plot
    ax2.plot(history.history['loss'], label='Train Loss', color='royalblue', marker='o')
    ax2.plot(history.history['val_loss'], label='Val Loss', color='orange', marker='s')
    ax2.set_title('Training & Validation Loss')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.set_xticks(range(len(history.history['loss'])))
    ax2.set_xticklabels(range(1, len(history.history['loss']) + 1))
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()


def main():
    data = read_data("dataset/07_bank_marketing_train.csv")
    X_train, X_test, y_train, y_test, balance_scaler, campaign_scaler, pdays_scaler, previous_scaler, age_scaler = preprocess(data)
    X_col = X_train.shape[1]
    y_col = 1
    print(f"size: {X_col}, {y_col}")

    print(X_train)
    print(X_test)
    print(y_train)
    print(y_test)
    model = create_model(X_col, y_col)
    model.summary()
    model_history = model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test), batch_size=64)

    plot_training_history(model_history)

    print("Hello, world\n")


if __name__=="__main__":
    main()
