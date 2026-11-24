# doodle

A collection of QuickDraw drawing classifiers implemented with scikit-learn,
TensorFlow/Keras, and VGG16 transfer learning.

## Dataset

Download the QuickDraw tutorial dataset:

http://download.tensorflow.org/data/quickdraw_tutorial_dataset_v1.tar.gz

Extract the archive under `dataset/`. The scripts accept either:

- `dataset/<dataset-name>/training.tfrecord-*` and
  `dataset/<dataset-name>/training.tfrecord.classes`, or
- one `.npy` file per class in `dataset/`. Each array should contain 784
  bitmap features or 800 legacy QuickDraw features.

The TFRecord loader rasterizes delta-encoded strokes to 28-by-28 bitmaps. Data
is split by class into 80% train, 10% development, and 10% test sets. The
random seed is fixed for reproducible splits.

## Setup

Use Python 3.11 or 3.12 and create a virtual environment:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

TensorFlow is required for TFRecord loading and neural-network training.
TensorBoard and a GPU-enabled TensorFlow installation are optional.

All commands below can be run from the repository root. Use `--help` to view
the options for a specific script.

## Shared data options

The classifier scripts support these common options:

- `--ip-folder`: dataset directory (default: `dataset/`)
- `--op-folder`: output directory (default: `results/`)
- `--classes`: class names to train; omit it to load every available class
- `--max-examples`: maximum examples per class (default: `100`; `0` loads all
  available examples)
- `--experiment-name`: optional output suffix for neural-network runs

Quote class names that contain spaces:

```sh
python -m Classifications.cnn --classes banana "paint can" "The Mona Lisa"
```

## Train the SVM classifier

Extract the QuickDraw archive into `dataset/`. The classifier reads its
`training.tfrecord-*` shards and class labels directly, rasterizing each
delta-encoded drawing to a 28-by-28 bitmap. For custom data, it also accepts
one NumPy file per class containing 784 bitmap or legacy 800-feature vectors.
Then run:

```sh
python -m Classifications.svm
```

The script evaluates a per-class train/dev/test split and saves confusion
matrices and plots to `results/`. It uses scikit-learn's `SVC` with a linear
kernel and up to 100 examples per class by default to limit training time.
Use `--classes` to select class names and `--max-examples` to change the
per-class sample limit (`0` uses all available examples). A 70,000-example
limit gives 7,000 dev examples per class when enough data is available, as in
the referenced confusion matrices. For example:

```sh
python -m Classifications.svm --classes banana eraser belt --max-examples 70000
python -m Classifications.svm --classes laptop "paint can" wristwatch rain panda "The Mona Lisa" nose pond "hockey stick" banana --max-examples 70000
```

Use `--kernel` to select another kernel (or `all` to compare them), and
`--gamma` to configure the RBF kernel (`scale`, `auto`, or a non-negative
number; defaults to `scale`). For example:

```sh
python -m Classifications.svm --classes banana eraser belt --max-examples 70000 --kernel rbf --gamma scale
```

Use `--help` for all options.

## Train the RNN

The RNN module is in `RNN/rnn.py`. It includes SimpleRNN and bidirectional
GRU architectures, plus convolutional-GRU variants:

```sh
python -m RNN.rnn --classes banana eraser belt --max-examples 500 --max-epochs 20 --model v6 --experiment-name three-class-gru
```

Use `--model all` to train every architecture, or select `v1` through `v7`.
Enable TensorBoard logging with `--tensorboard`. Results include a model,
weights, class names, confusion matrix, and text report under `results/`.

Use `--help` for all options.

## Train the random forest classifier

The random forest uses the same QuickDraw dataset and train/dev/test split:

```sh
python -m Classifications.randomforest --classes banana eraser belt --max-examples 500
```

Use `--n-estimators` to set the number of trees and `--max-depth` to limit
their depth. The default is 100 trees with unlimited depth. The classifier
reports dev accuracy and writes the class names and confusion matrix to
`results/`. Use `--help` for all options.

## Train the CNN

The CNN uses the same QuickDraw training TFRecord data and 28-by-28 bitmap
conversion. To try a small three-class run:

```sh
python -m Classifications.cnn --classes banana eraser belt --max-examples 500 --max-epochs 20 --experiment-name three-class
```

To train across every class in the dataset, omit `--classes`. For a larger
run, increase the per-class sample limit:

```sh
python -m Classifications.cnn --max-examples 500 --max-epochs 100 --experiment-name all-classes
```

The default is 100 examples per class and 100 epochs. `--max-examples 0`
loads all available examples, which can require substantial memory. The data
loader makes per-class train/dev/test splits from the training TFRecord files;
the test score is therefore a held-out portion of those files, not a score on
the dataset's separate official evaluation TFRecords.

The CNN reports test loss and accuracy and writes the model, weights, class
names, confusion matrix, and report to `results/`. The experiment name is
appended to output filenames so runs can be kept separate. TensorBoard is
optional and can be enabled with `--tensorboard` when it is installed and
working in the active Python environment. Use `--help` for all options.

## Train the VGG16 transfer-learning classifier

The VGG16 script resizes the 28-by-28 drawings to 224-by-224 RGB images and
applies VGG16 preprocessing. It first trains a classifier head over frozen
ImageNet-pretrained VGG16 features, then fine-tunes the final four VGG16 layers
at a lower learning rate. For a small run:

```sh
python -m Classifications.vgg --classes banana eraser belt --max-examples 100 --max-epochs 5 --fine-tune-epochs 5 --experiment-name three-class
```

Use `--fine-tune-layers` to change how many of the final VGG16 layers are
unfrozen, `--tensorboard` to enable TensorBoard logs, and `--help` for all
options. The script saves separate models and confusion matrices for the
transfer-learning and fine-tuned phases under `results/`. The first run may
download the pretrained VGG16 weights if they are not already cached.

## Train logistic regression

The logistic-regression baseline uses unscaled bitmap features, the `lbfgs`
solver, and 100 iterations by default, matching the linked reference setup.
Select the same classes and sample limit to compare runs:

```sh
python -m Classifications.logs --classes banana eraser belt --max-examples 70000
```

It reports dev and test accuracy and saves a dev-set confusion matrix to
`results/`. Use `--max-iter` to change the solver iteration limit.

## Project structure

- `Classifications/data.py`: dataset loading and train/dev/test splitting
- `Classifications/svm.py`: linear, polynomial, RBF, and sigmoid SVM classifiers
- `Classifications/randomforest.py`: random forest classifier
- `Classifications/logs.py`: logistic-regression baseline
- `Classifications/cnn.py`: convolutional neural network
- `Classifications/vgg.py`: VGG16 transfer learning and fine-tuning
- `RNN/rnn.py`: SimpleRNN, GRU, and convolutional-GRU architectures
- `Classifications/utils.py`: labels and plotting helpers
- `dataset/`: downloaded QuickDraw TFRecord or NumPy data
- `results/`: trained models, weights, reports, plots, and TensorBoard logs

## Results

For the existing three-class examples, the best recorded results are SVM RBF
with gamma `scale` at 67% and the base CNN example at 84%. Results vary with
class selection, examples per class, random split, and training configuration.
