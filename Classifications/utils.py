import matplotlib.pyplot as plt
import numpy as np

bitmap_dim = 800

# confusion matrix and graph plotting 
def plot_confusion_matrix(
    matrix,
    class_names,
    normalize=False,
    title="Confusion matrix",
    ax=None,
):
    
    matrix = np.asarray(matrix)
    class_names = list(class_names)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be a square, two-dimensional array")
    if matrix.shape[0] != len(class_names):
        raise ValueError("class_names must contain one name per matrix row")

    if normalize:
        row_totals = matrix.sum(axis=1, keepdims=True)
        matrix = np.divide(
            matrix,
            row_totals,
            out=np.zeros_like(matrix, dtype=float),
            where=row_totals != 0,
        )

    if ax is None:
        _, ax = plt.subplots()

    image = ax.imshow(matrix, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(image, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    threshold = matrix.max() / 2 if matrix.size else 0
    value_format = ".2f" if normalize else "g"
    for row, column in np.ndindex(matrix.shape):
        ax.text(
            column,
            row,
            format(matrix[row, column], value_format),
            ha="center",
            va="center",
            color="white" if matrix[row, column] > threshold else "black",
        )

    ax.figure.tight_layout()
    return ax

def get_labels(Y): 
    labels = list(set(Y))
    return labels