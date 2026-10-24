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
matrices and plots to `results/`. It uses a linear SVM and up to 100 examples
per class by default to limit training time. Use `--kernel` to select another
kernel (or `all` to compare them), `--max-examples` to change the sample limit,
and `--help` for all options.