# -*- coding: utf-8 -*-


import argparse
import json
from pathlib import Path

import numpy as np
import scipy.io as sc
import tensorflow as tf
import keras

from model_builder import build_model


def load_data(data_path):

    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    # Both supported file formats use the same signal and label keys.
    if data_path.suffix.lower() == ".npz":
        with np.load(data_path) as data:
            Train_signal_data = data["Train_signal_data"]
            label_data_train = data["label_data_train"]
    elif data_path.suffix.lower() == ".mat":
        data = sc.loadmat(data_path)
        Train_signal_data = data["Train_signal_data"]
        label_data_train = data["label_data_train"]
    else:
        raise ValueError("The data file must be in .npz or .mat format.")

    return np.asarray(Train_signal_data), np.asarray(label_data_train)


def train_model(Train_signal_data, label_data_train, history_path="history.json"):

    # Expect two-channel signals and one-hot labels with matching sample counts.
    if Train_signal_data.ndim != 3 or Train_signal_data.shape[-1] != 2:
        raise ValueError(
            "Train_signal_data must have shape [num_samples, signal_length, 2]."
        )
    if label_data_train.ndim != 2:
        raise ValueError(
            "label_data_train must be a two-dimensional one-hot label matrix "
            "with shape [num_samples, num_classes]."
        )
    if Train_signal_data.shape[0] != label_data_train.shape[0]:
        raise ValueError("The number of training signals must match the number of labels.")

    Len_signal = Train_signal_data.shape[1]
    class_num = label_data_train.shape[1]

    model = build_model(Len_signal, class_num)
    model.summary()


    # Use categorical cross-entropy for the one-hot classification targets.
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    BATCH_SIZE = 128
    EPOCH = 50

    # Reserve 30% of the data for validation and stop after five unimproved epochs.
    hist = model.fit(
        [Train_signal_data],
        label_data_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCH,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=5,
                verbose=0,
                mode="auto",
            )
        ],
        validation_split=0.3,
    )


    # Convert metric values to native Python types for JSON serialization.
    history = {
        "epoch": [int(epoch) for epoch in hist.epoch],
        "loss": [float(v) for v in hist.history["loss"]],
        "val_loss": [float(v) for v in hist.history["val_loss"]],
        "accuracy": [float(v) for v in hist.history["accuracy"]],
        "val_accuracy": [float(v) for v in hist.history["val_accuracy"]],
    }
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print(f"Training complete. Training history saved to: {history_path}")
    return model, hist


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the SpikingScattering model")
    parser.add_argument(
        "--data",
        required=True,
        help="A .npz or .mat file containing Train_signal_data and label_data_train",
    )
    parser.add_argument("--history", default="history.json", help="Output path for the training history")
    args = parser.parse_args()

    X_train, y_train = load_data(args.data)
    train_model(X_train, y_train, history_path=args.history)
