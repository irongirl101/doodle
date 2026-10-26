# doodle

Dataset 
http://download.tensorflow.org/data/quickdraw_tutorial_dataset_v1.tar.gz

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
`--help` for all options.

## Train logistic regression

The logistic-regression baseline uses unscaled bitmap features, the `lbfgs`
solver, and 100 iterations by default, matching the linked reference setup.
Select the same classes and sample limit to compare runs:

```sh
python -m Classifications.logs --classes banana eraser belt --max-examples 70000
```

It reports dev and test accuracy and saves a dev-set confusion matrix to
`results/`. Use `--max-iter` to change the solver iteration limit.