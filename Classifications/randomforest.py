"""Train a random forest classifier on the QuickDraw dataset."""

import argparse
import pickle
import time
from pathlib import Path

if __package__:
    from Classifications import data, utils
else:
    import data
    import utils

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def train(data_get, output_folder, n_estimators=100, max_depth=None, n_jobs=-1):
    X_train, y_train = data_get.get_train()
    X_dev, y_dev = data_get.get_dev()

    classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=n_jobs,
        random_state=42,
    )
    start_secs = time.perf_counter()
    classifier.fit(X_train, y_train)
    duration = time.perf_counter() - start_secs
    print(f"Trained random forest in {duration:.2f} seconds")

    dev_predictions = classifier.predict(X_dev)
    accuracy = classifier.score(X_dev, y_dev)
    print(f"Dev accuracy: {accuracy:.4f}")

    class_names = sorted(utils.get_labels(y_dev))
    confusion = confusion_matrix(y_dev, dev_predictions, labels=class_names)
    print("Confusion matrix:", confusion)

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    filename = "randomforest"
    with (output_folder / f"{filename}_classes.pkl").open("wb") as file:
        pickle.dump(class_names, file)
    with (output_folder / f"{filename}_confusion.pkl").open("wb") as file:
        pickle.dump(confusion, file)

    ax = utils.plot_confusion_matrix(confusion, class_names, title=filename)
    ax.figure.savefig(
        output_folder / f"{filename}_confusion.png",
        bbox_inches="tight",
    )
    import matplotlib.pyplot as plt

    plt.close(ax.figure)


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
        "--n-estimators",
        type=int,
        default=100,
        help="Number of trees in the forest (default: 100)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        help="Maximum depth of each tree (default: unlimited)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel jobs for fitting and prediction (default: all CPUs)",
    )
    options = parser.parse_args()
    if options.max_examples < 0:
        parser.error("--max-examples must be zero or greater")
    if options.n_estimators < 1:
        parser.error("--n-estimators must be at least 1")
    if options.max_depth is not None and options.max_depth < 1:
        parser.error("--max-depth must be at least 1")
    if options.n_jobs == 0:
        parser.error("--n-jobs cannot be zero")

    data_get = data.messagedata(
        folder=options.ip_folder,
        max_examples=options.max_examples,
        classes=options.classes,
    )
    train(
        data_get,
        options.op_folder,
        n_estimators=options.n_estimators,
        max_depth=options.max_depth,
        n_jobs=options.n_jobs,
    )


if __name__ == "__main__":
    main()
