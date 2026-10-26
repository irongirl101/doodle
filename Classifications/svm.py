"""Train several SVM kernels on the QuickDraw dataset."""

import argparse
import pickle
import time
from pathlib import Path

if __package__:
    from Classifications import data, utils
else:
    import data
    import utils

from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def kernel(data_get, output_folder, **args):
    print(args)
    X_train,y_train = data_get.get_train()
    X_dev,y_dev = data_get.get_dev()

    # Fit scaling on training data only to avoid data leakage
    scalin = MinMaxScaler(feature_range=(-1,1)).fit(X_train)
    X_train = scalin.transform(X_train)
    X_dev = scalin.transform(X_dev)

    start_secs = time.perf_counter()
    kernel_name = args.pop("kernel")
    clf = svm.SVC(kernel=kernel_name, **args)
    clf.fit(X_train,y_train)
    dur = time.perf_counter() - start_secs
    print('Trained, took',dur,'s')

    dev_pred = clf.predict(X_dev)
    accuracy = clf.score(X_dev,y_dev)
    print("accuracy: ",accuracy)

    class_names = sorted(utils.get_labels(y_dev))
    confusion = confusion_matrix(y_dev,dev_pred,labels=class_names)
    print('Confusion matrix',confusion)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    filename = f"svm_{kernel_name}"
    with (output_folder / f"{filename}_classes.pkl").open("wb") as file:
        pickle.dump(class_names, file)
    with (output_folder / f"{filename}_confusion.pkl").open("wb") as file:
        pickle.dump(confusion, file)
    ax = utils.plot_confusion_matrix(confusion, class_names, title=filename)
    ax.figure.savefig(output_folder / f"{filename}_confusion.png", bbox_inches="tight")
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
        "--kernel",
        choices=("linear", "poly", "rbf", "sigmoid", "all"),
        default="rbf",
        help="SVM kernel to train (default: linear; 'all' runs every kernel)",
    )
    options = parser.parse_args()
    if options.max_examples < 0:
        parser.error("--max-examples must be zero or greater")

    data_get = data.messagedata(
        folder=options.ip_folder,
        max_examples=options.max_examples,
        classes=options.classes,
    )
    kernels = {
        "linear": {},
        "poly": {"degree": 5, "coef0": 1},
        "rbf": {"gamma": 1},
        "sigmoid": {"coef0": 1},
    }
    selected_kernels = kernels if options.kernel == "all" else {
        options.kernel: kernels[options.kernel]
    }
    for kernel_name, parameters in selected_kernels.items():
        kernel(
            data_get,
            options.op_folder,
            kernel=kernel_name,
            verbose=1,
            max_iter=500,
            **parameters,
        )


if __name__ == "__main__":
    main()