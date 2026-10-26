"""Train and evaluate a logistic-regression classifier on QuickDraw."""

import argparse
import pickle
import time
from pathlib import Path

if __package__:
    from Classifications import data, utils
else:
    import data
    import utils

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ip-folder",
        type=Path,
        default=PROJECT_ROOT / "dataset",
        help="Dataset directory (default: repository dataset/)",
    )
    parser.add_argument(
        "--op-folder",
        type=Path,
        default=PROJECT_ROOT / "results",
        help="Results directory (default: repository results/)",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=100,
        help="Maximum examples per class (0 means all)",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        help="Class names to load (default: every class in the dataset)",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=100,
        help="Maximum solver iterations (default: 100, matching the reference)",
    )
    options = parser.parse_args()
    if options.max_examples < 0:
        parser.error("--max-examples must be zero or greater")
    if options.max_iter <= 0:
        parser.error("--max-iter must be greater than zero")

    dataset = data.messagedata(
        folder=options.ip_folder,
        max_examples=options.max_examples,
        classes=options.classes,
    )
    X_train, y_train = dataset.get_train()
    X_dev, y_dev = dataset.get_dev()
    X_test, y_test = dataset.get_test()

    classifier = LogisticRegression(
        random_state=0,
        solver="lbfgs",
        max_iter=options.max_iter,
        verbose=1,
    )
    start_time = time.perf_counter()
    classifier.fit(X_train, y_train)
    training_time = time.perf_counter() - start_time

    dev_predictions = classifier.predict(X_dev)
    print(f"Training completed in {training_time:.1f} seconds")
    print(f"dev accuracy: {classifier.score(X_dev, y_dev):.4f}")
    print(f"test accuracy: {classifier.score(X_test, y_test):.4f}")

    class_names = classifier.classes_.tolist()
    dev_confusion = confusion_matrix(y_dev, dev_predictions, labels=class_names)
    output_folder = Path(options.op_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    with (output_folder / "logistic_regression.pkl").open("wb") as model_file:
        pickle.dump(classifier, model_file)
    with (output_folder / "logistic_regression_confusion.pkl").open("wb") as matrix_file:
        pickle.dump(dev_confusion, matrix_file)

    ax = utils.plot_confusion_matrix(
        dev_confusion,
        class_names,
        title="Logistic regression dev confusion matrix",
    )
    ax.figure.savefig(
        output_folder / "logistic_regression_confusion.png",
        bbox_inches="tight",
    )
    import matplotlib.pyplot as plt

    plt.close(ax.figure)


if __name__ == "__main__":
    main()
