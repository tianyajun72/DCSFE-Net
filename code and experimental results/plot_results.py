# -*- coding: utf-8 -*-


import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# Share the same typography across the confusion-matrix axis labels.
font2 = {
    "family": "Times New Roman",
    "weight": "bold",
    "size": 26,
}


def plot_confusion_matrix(cm, title="Confusion matrix", cmap=plt.cm.Blues, labels=None):

    if labels is None:
        labels = []

    plt.figure()
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=20)

    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.xticks(fontsize=24, family="Times New Roman")
    plt.yticks(tick_marks, labels)
    plt.yticks(fontsize=24, family="Times New Roman")
    plt.ylabel("True label", font2)
    plt.xlabel("Predicted label", font2)

    # Display the sample count in each confusion-matrix cell.
    for first_index in range(len(labels)):
        for second_index in range(len(labels)):
            plt.text(
                first_index - 0.4,
                second_index + 0.1,
                int(cm[second_index][first_index]),
                fontsize=24,
                family="Times New Roman",
            )


def plot_training_curves(history_path="history.json", output_dir="."):

    history_path = Path(history_path)
    if not history_path.exists():
        raise FileNotFoundError(
            f"Training history not found: {history_path}. Run train_model.py first to complete training."
        )

    # Load the metrics exported by train_model.py.
    with open(history_path, "r", encoding="utf-8") as f:
        hist = json.load(f)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = hist["epoch"]

    plt.figure()
    plt.title("Training performance")
    # Compare training and validation loss over the recorded epochs.
    plt.plot(epochs, hist["loss"], label="train loss+error")
    plt.plot(epochs, hist["val_loss"], label="val_error")
    plt.legend()
    plt.savefig(output_dir / "training_loss.png", dpi=300, bbox_inches="tight")

    plt.figure()
    plt.title("Training performance")
    # Save the accuracy comparison as a separate figure.
    plt.plot(epochs, hist["accuracy"], label="train accuracy+error")
    plt.plot(epochs, hist["val_accuracy"], label="val_error")
    plt.legend()
    plt.savefig(output_dir / "training_accuracy.png", dpi=300, bbox_inches="tight")

    print(f"Training loss plot: {output_dir / 'training_loss.png'}")
    print(f"Training accuracy plot: {output_dir / 'training_accuracy.png'}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training results")
    parser.add_argument("--history", default="history.json", help="Training history JSON file")
    parser.add_argument("--output-dir", default=".", help="Directory for saving plots")
    args = parser.parse_args()
    plot_training_curves(args.history, args.output_dir)
