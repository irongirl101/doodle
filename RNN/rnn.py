# My actual part of the project :smiling_imp: 

# RNNs process sequential data, by retaining info from previous steps. 
# Probably should use a LSTM /GRU 

import argparse
import datetime
import pickle
import time
from pathlib import Path

import numpy as np 
import tensorflow as tf 
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense
import keras 
from sklearn.preprocessing import LabelEncoder

if __package__:
    from Classifications import data, utils
else:
    import Classifications.data
    import Classifications.utils


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EPOCHS = 100 
BATCH_SIZE = 500 

model_file = 'simple_rnn_model'
model_weights_file = 'simple_rnn_model_weights'
class_file = 'simple_rnn_classes'
confusion_file = 'simple_rnn_confusion'

def get_data_folder(options):
    return Path(options.ip_folder)


def get_num_classes(dataset):
    _, labels = dataset.get_data()
    return len(np.unique(labels))


def get_suffix_name(options):
    return f"_{options.experiment_name}" if options.experiment_name else ""


def get_class_filename(options):
    return Path(options.op_folder) / f"{class_file}{get_suffix_name(options)}.pkl"


def get_confusion_matrix_name(options):
    return f"{confusion_file}{get_suffix_name(options)}"


def get_confusion_matrix_filename(options):
    return Path(options.op_folder) / f"{get_confusion_matrix_name(options)}.pkl"


def get_model_filename(options):
    return Path(options.op_folder) / f"{model_file}{get_suffix_name(options)}.keras"

def get_model_name_only(options):
    if options.model == "v1":
        return f"{model_file}{get_suffix_name(options)}"
    return f"{model_file}_{options.model}{get_suffix_name(options)}"


def get_tensorboard_directory(options):
    run_directory = (
        Path(options.op_folder)
        / "tensorboard"
        / get_model_name_only(options)
        / datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    run_directory.mkdir(parents=True, exist_ok=True)
    return run_directory


def get_training_plots_directory(options):
    directory = (
        Path(options.op_folder)
        / "training_plots"
        / get_model_name_only(options)
    )
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def get_training_plot_filename(options):
    return (
        get_training_plots_directory(options)
        / f"{get_model_name_only(options)}_training_plot"
    )


def get_tensorboard_callback(options, frequency=2):
    log_directory = get_tensorboard_directory(options)
    print(f"TensorBoard log directory: {log_directory}")
    return keras.callbacks.TensorBoard(
        log_dir=str(log_directory),
        histogram_freq=frequency,
        write_graph=True,
        write_images=True,
    )


def get_model_weights_filename(options):
    return Path(options.op_folder) / (
        f"{model_weights_file}{get_suffix_name(options)}.weights.h5"
    )


def get_experiment_report_filename(options):
    return Path(options.op_folder) / (
        f"simple_rnn_keras_results{get_suffix_name(options)}.txt"
    )

def encode_values(encoder, Y, confoose=False):
    encoded = encoder.fit_transform(Y)
    if confoose:
        return encoded
    return keras.utils.to_categorical(encoded, num_classes=len(encoder.classes_))

def decode(encoder, y):
    return encoder.inverse_transform(y)

def model_v1(num_classes):
    model = Sequential([
        keras.Input(shape=(28, 28)),
        SimpleRNN(50, activation="relu"),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


"""def model_v2(num_classes):
    model = Sequential([
        keras.Input(shape=(28, 28)),
        SimpleRNN(64, activation="relu", return_sequences=True),
        SimpleRNN(32, activation="relu"),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def model_v3(num_classes):
    model = Sequential([
        keras.Input(shape=(28, 28)),
        SimpleRNN(128, activation="relu"),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def model_v4(num_classes):
    model = Sequential([
        keras.Input(shape=(28, 28)),
        SimpleRNN(64, activation="relu", return_sequences=True),
        SimpleRNN(64, activation="relu"),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def model_v5(num_classes):
    model = Sequential([
        keras.Input(shape=(28, 28)),
        SimpleRNN(32, activation="relu"),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model"""


MODEL_VARIANTS = {
    "v1": model_v1,
   # "v2": model_v2,
   # "v3": model_v3,
    #"v4": model_v4,
    #"v5": model_v5,
}

def one_hot(index,num_classes): 
    result = np.zeros(num_classes)
    result[index] = 1
    return result

def convert_predictions_to_onehot(predictions, num_classes):
    predictions = np.asarray(predictions, dtype=int)
    return np.eye(num_classes)[predictions]

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
        "--max-epochs",
        type=int,
        default=100,
        help="Maximum epochs",
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
        "--experiment-name",
        default="",
        help="Optional suffix used to keep this run's output files separate",
    )
    parser.add_argument(
        "--model",
        choices=("v1", "v2", "v3", "v4", "v5", "all"),
        default="v1",
        help="CNN architecture to train (default: v1; 'all' runs every architecture)",
    )
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="Enable TensorBoard logging (requires a working TensorBoard install)",
    )
    options = parser.parse_args()
    if options.max_epochs <= 0:
        parser.error("--max-epochs must be greater than zero")
    if options.max_examples < 0:
        parser.error("--max-examples must be zero or greater")

    dataset = data.messagedata(
        folder=get_data_folder(options),
        max_examples=options.max_examples,
        classes=options.classes,
    )
    X_train, y_train = dataset.get_train()
    X_dev, y_dev = dataset.get_dev()
    X_test, y_test = dataset.get_test()

    def prepare_images(features):
        features = np.asarray(features, dtype=np.float32)
        if features.ndim != 2 or features.shape[1] != 28 * 28:
            raise ValueError("RNN requires flattened 28x28 bitmap data (784 features)")
        if features.size and features.max() > 1:
            features = features / 255.0
        return features.reshape((-1, 28, 28))

    encoder = LabelEncoder()
    y_train = encode_values(encoder, y_train, confoose=True)
    y_dev = encoder.transform(y_dev)
    y_test = encoder.transform(y_test)

    X_train = prepare_images(X_train)
    X_dev = prepare_images(X_dev)
    X_test = prepare_images(X_test)
    model_names = list(MODEL_VARIANTS) if options.model == "all" else [options.model]
    output_folder = Path(options.op_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    for model_name in model_names:
        options.model = model_name
        classifier = MODEL_VARIANTS[model_name](len(encoder.classes_))
        start_time = time.perf_counter()
        history = classifier.fit(
            X_train,
            keras.utils.to_categorical(y_train, num_classes=len(encoder.classes_)),
            validation_data=(
                X_dev,
                keras.utils.to_categorical(y_dev, num_classes=len(encoder.classes_)),
            ),
            epochs=options.max_epochs,
            callbacks=[get_tensorboard_callback(options)] if options.tensorboard else [],
            verbose=2,
        )
        training_time = time.perf_counter() - start_time
        test_loss, test_accuracy = classifier.evaluate(
            X_test,
            keras.utils.to_categorical(y_test, num_classes=len(encoder.classes_)),
            verbose=0,
        )
        predictions = np.argmax(classifier.predict(X_test, verbose=0), axis=1)
        class_names = encoder.classes_.tolist()
        matrix = confusion_matrix(
            y_test,
            predictions,
            labels=np.arange(len(class_names)),
        )
        classifier.save(get_model_filename(options))
        classifier.save_weights(get_model_weights_filename(options))
        with get_class_filename(options).open("wb") as file:
            pickle.dump(class_names, file)
        with get_confusion_matrix_filename(options).open("wb") as file:
            pickle.dump(matrix, file)

    ax = utils.plot_confusion_matrix(
            matrix, class_names, title=f"RNN {model_name} test confusion matrix"
        )
    ax.figure.savefig(
            output_folder / f"{get_confusion_matrix_name(options)}.png",
            bbox_inches="tight",
    )
    import matplotlib.pyplot as plt

    plt.close(ax.figure)
    report = (
            f"Model: {model_name}\n"
            f"Training time: {training_time:.1f} seconds\n"
            f"Test loss: {test_loss:.4f}\n"
            f"Test accuracy: {test_accuracy:.4f}\n"
        )
    get_experiment_report_filename(options).write_text(report, encoding="utf-8")
    print(report, end="")

    options.model = "v1"


if __name__ == "__main__":
    main()