#logistic regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
if __package__:
    from Classifications import data, utils
else:
    import data
    import utils
from sklearn.metrics import confusion_matrix
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler


dataset_dir = Path(__file__).resolve().parents[1]
dataset = data.messagedata(folder=dataset_dir,max_examples=100)

X_train,y_train = dataset.get_train() 
X_dev,y_dev = dataset.get_dev()
X_test,y_test = dataset.get_test()

scalin = MinMaxScaler(feature_range=(-1,1)).fit(X_train)
X_train = scalin.transform(X_train)
X_dev = scalin.transform(X_dev)
X_test = scalin.transform(X_test)

import pickle
import time

start_time = time.perf_counter()
classifier = LogisticRegression(max_iter=500, solver="lbfgs")
classifier.fit(X_train, y_train)
print(f"Training completed in {time.perf_counter() - start_time:.1f} seconds")

dev_predictions = classifier.predict(X_dev)
test_predictions = classifier.predict(X_test)
print(f"dev accuracy: {classifier.score(X_dev, y_dev):.4f}")
print(f"test accuracy: {classifier.score(X_test, y_test):.4f}")

class_names = classifier.classes_.tolist()
test_confusion = confusion_matrix(y_test, test_predictions, labels=class_names)

results_dir = dataset_dir / "results"
results_dir.mkdir(parents=True, exist_ok=True)
with (results_dir / "logistic_regression.pkl").open("wb") as model_file:
    pickle.dump({"classifier": classifier, "scaler": scalin}, model_file)
with (results_dir / "logistic_regression_confusion.pkl").open("wb") as matrix_file:
    pickle.dump(test_confusion, matrix_file)

ax = utils.plot_confusion_matrix(
    test_confusion,
    class_names,
    title="Logistic regression test confusion matrix",
)
ax.figure.savefig(
    results_dir / "logistic_regression_confusion.png",
    bbox_inches="tight",
)
import matplotlib.pyplot as plt

plt.close(ax.figure)
