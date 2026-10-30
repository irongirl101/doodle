"""Train a VGG16 QuickDraw classifier with transfer learning and fine-tuning."""

import argparse
import datetime
import pickle
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from keras.models import Model

if __package__:
    from Classifications import data, utils
else:
    import data
    import utils


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_SIZE = 224
BATCH_SIZE = 32
MODEL_FILE = "vgg16_model"
CLASS_FILE = "vgg16_classes"
CONFUSION_FILE = "vgg16_confusion"
REPORT_FILE = "vgg16_results"


def get_suffix_name(options):
    return f"_{options.experiment_name}" if options.experiment_name else ""


def get_output_stem(options, phase):
    return f"{MODEL_FILE}_{phase}{get_suffix_name(options)}"


def make_dataset(features, labels, batch_size, training=False):
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    if training:
        dataset = dataset.shuffle(
            buffer_size=len(features),
            seed=42,
            reshuffle_each_iteration=True,
        )

    def prepare_image(image, label):
        image = tf.cast(tf.reshape(image, (28, 28, 1)), tf.float32)
        image = tf.cond(
            tf.reduce_max(image) > 1.0,
            lambda: image,
            lambda: image * 255.0,
        )
        image = tf.image.grayscale_to_rgb(image)
        image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        return preprocess_input(image), label

    return (
        dataset.map(prepare_image, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )


def build_model(num_classes):
    base_model = VGG16(
        include_top=False,
        weights="imagenet",
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
    )
    base_model.trainable = False

    inputs = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    features = base_model(inputs, training=False)
    features = GlobalAveragePooling2D()(features)
    features = Dense(256, activation="relu")(features)
    features = Dropout(0.5)(features)
    outputs = Dense(num_classes, activation="softmax")(features)
    model = Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, base_model


def get_tensorboard_callback(options, phase):
    log_directory = (
        Path(options.op_folder)
        / "tensorboard"
        / get_output_stem(options, phase)
        / datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    log_directory.mkdir(parents=True, exist_ok=True)
    print(f"TensorBoard log directory: {log_directory}")
    return tf.keras.callbacks.TensorBoard(log_dir=str(log_directory))


def save_phase_results(model, options, phase, test_dataset, y_test, class_names):
    test_loss, test_accuracy = model.evaluate(test_dataset, verbose=0)
    predictions = np.argmax(model.predict(test_dataset, verbose=0), axis=1)
    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=np.arange(len(class_names)),
    )

    output_folder = Path(options.op_folder)
    output_stem = get_output_stem(options, phase)
    model.save(output_folder / f"{output_stem}.keras")
    model.save_weights(output_folder / f"{output_stem}.weights.h5")
    with (output_folder / f"{CLASS_FILE}{get_suffix_name(options)}.pkl").open(
        "wb"
    ) as file:
        pickle.dump(class_names, file)
    with (
        output_folder
        / f"{CONFUSION_FILE}_{phase}{get_suffix_name(options)}.pkl"
    ).open("wb") as file:
        pickle.dump(matrix, file)

    ax = utils.plot_confusion_matrix(
        matrix,
        class_names,
        title=f"VGG16 {phase.replace('_', ' ')} test confusion matrix",
    )
    ax.figure.savefig(
        output_folder / f"{CONFUSION_FILE}_{phase}{get_suffix_name(options)}.png",
        bbox_inches="tight",
    )
    import matplotlib.pyplot as plt

    plt.close(ax.figure)
    return test_loss, test_accuracy


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
        default=10,
        help="Epochs for training the new classifier head",
    )
    parser.add_argument(
        "--fine-tune-epochs",
        type=int,
        default=10,
        help="Epochs for fine-tuning the final VGG16 layers",
    )
    parser.add_argument(
        "--fine-tune-layers",
        type=int,
        default=4,
        help="Number of final VGG16 layers to unfreeze during fine-tuning",
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
        "--tensorboard",
        action="store_true",
        help="Enable TensorBoard logging",
    )
    options = parser.parse_args()
    if options.max_epochs <= 0:
        parser.error("--max-epochs must be greater than zero")
    if options.fine_tune_epochs <= 0:
        parser.error("--fine-tune-epochs must be greater than zero")
    if options.fine_tune_layers <= 0:
        parser.error("--fine-tune-layers must be greater than zero")
    if options.max_examples < 0:
        parser.error("--max-examples must be zero or greater")

    dataset = data.messagedata(
        folder=options.ip_folder,
        max_examples=options.max_examples,
        classes=options.classes,
    )
    X_train, y_train = dataset.get_train()
    X_dev, y_dev = dataset.get_dev()
    X_test, y_test = dataset.get_test()

    for features, split_name in (
        (X_train, "training"),
        (X_dev, "development"),
        (X_test, "test"),
    ):
        if np.asarray(features).ndim != 2 or np.asarray(features).shape[1] != 784:
            raise ValueError(
                f"VGG16 requires flattened 28x28 bitmap data; invalid {split_name} data"
            )

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train)
    y_dev = encoder.transform(y_dev)
    y_test = encoder.transform(y_test)
    class_names = encoder.classes_.tolist()

    train_dataset = make_dataset(X_train, y_train, BATCH_SIZE, training=True)
    dev_dataset = make_dataset(X_dev, y_dev, BATCH_SIZE)
    test_dataset = make_dataset(X_test, y_test, BATCH_SIZE)

    output_folder = Path(options.op_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    model, base_model = build_model(len(class_names))
    if options.fine_tune_layers > len(base_model.layers):
        parser.error(
            f"--fine-tune-layers cannot exceed {len(base_model.layers)} "
            "VGG16 base layers"
        )
    reports = []

    start_time = time.perf_counter()
    model.fit(
        train_dataset,
        validation_data=dev_dataset,
        epochs=options.max_epochs,
        callbacks=(
            [get_tensorboard_callback(options, "transfer_learning")]
            if options.tensorboard
            else []
        ),
        verbose=2,
    )
    training_time = time.perf_counter() - start_time
    test_loss, test_accuracy = save_phase_results(
        model, options, "transfer_learning", test_dataset, y_test, class_names
    )
    reports.append(
        f"Phase: transfer learning\n"
        f"Training time: {training_time:.1f} seconds\n"
        f"Test loss: {test_loss:.4f}\n"
        f"Test accuracy: {test_accuracy:.4f}\n"
    )

    base_model.trainable = True
    first_trainable_layer = len(base_model.layers) - options.fine_tune_layers
    for layer in base_model.layers[:first_trainable_layer]:
        layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    start_time = time.perf_counter()
    model.fit(
        train_dataset,
        validation_data=dev_dataset,
        epochs=options.fine_tune_epochs,
        callbacks=(
            [get_tensorboard_callback(options, "fine_tuned")]
            if options.tensorboard
            else []
        ),
        verbose=2,
    )
    training_time = time.perf_counter() - start_time
    test_loss, test_accuracy = save_phase_results(
        model, options, "fine_tuned", test_dataset, y_test, class_names
    )
    reports.append(
        f"Phase: fine-tuning ({options.fine_tune_layers} VGG16 layers unfrozen)\n"
        f"Training time: {training_time:.1f} seconds\n"
        f"Test loss: {test_loss:.4f}\n"
        f"Test accuracy: {test_accuracy:.4f}\n"
    )

    report = "\n".join(reports)
    report_path = output_folder / f"{REPORT_FILE}{get_suffix_name(options)}.txt"
    report_path.write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()