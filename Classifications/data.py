from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

BITMAP = 28 * 28
RANDOM_SEED = 42 

class messagedata: 
    def binarize(self,x): # this sets all the pixel values above 0 to be 225 -- so its black or white.
        x[x>0] = 225 
        return x 

    def get_data(self): 
        return(self.X,self.y)

    def get_train(self): 
        return(self.X_train,self.y_train)

    def get_test(self):
        return self.X_test, self.y_test

    def get_dev(self): 
        return(self.X_dev, self.y_dev)

    def __init__(self, folder='dataset', binarize=False, max_examples=500):
        self.folder = folder
        data_dir = Path(folder)
        npy_files = sorted(data_dir.glob("*.npy"))
        if npy_files:
            features, labels = self._load_npy_files(npy_files, max_examples)
        else:
            shards = sorted(data_dir.rglob("training.tfrecord-*"))
            if not shards:
                raise FileNotFoundError(
                    f"No .npy files or training TFRecord shards found in {data_dir}"
                )
            features, labels = self._load_tfrecords(shards, max_examples)

        if binarize:
            features = self.binarize(features)

        self.X, self.y = features, labels
        self.X_train, self.X_dev, self.X_test, self.y_train, self.y_dev, self.y_test = (
            self._split(features, labels)
        )

    @staticmethod
    def _load_npy_files(files, max_examples):
        features, labels = [], []
        for file_path in files:
            class_features = np.load(file_path, allow_pickle=False)
            if class_features.ndim != 2 or class_features.shape[1] not in (BITMAP, 800):
                raise ValueError(
                    f"{file_path} must contain arrays with {BITMAP} or 800 features"
                )
            if max_examples > 0:
                class_features = class_features[:max_examples]
            features.append(class_features)
            labels.append(np.full(len(class_features), file_path.stem))
        return np.concatenate(features), np.concatenate(labels)

    @staticmethod
    def _load_tfrecords(shards, max_examples):
        import tensorflow as tf

        classes_path = shards[0].parent / "training.tfrecord.classes"
        if not classes_path.is_file():
            raise FileNotFoundError(f"Missing class labels file: {classes_path}")
        class_names = classes_path.read_text(encoding="utf-8").splitlines()
        examples, counts = [[] for _ in class_names], np.zeros(len(class_names), dtype=int)

        for record in tf.data.TFRecordDataset([str(path) for path in shards]):
            example = tf.train.Example.FromString(record.numpy())
            record_features = example.features.feature
            class_index = record_features["class_index"].int64_list.value[0]
            if class_index < 0 or class_index >= len(class_names):
                raise ValueError(f"Invalid class index {class_index} in TFRecord")
            if max_examples > 0 and counts[class_index] >= max_examples:
                if np.all(counts >= max_examples):
                    break
                continue

            ink = np.asarray(record_features["ink"].float_list.value, dtype=np.float32)
            shape = tuple(record_features["shape"].int64_list.value)
            if len(shape) != 2 or shape[1] != 3 or np.prod(shape) != ink.size:
                raise ValueError(f"Invalid ink shape {shape} in TFRecord")
            examples[class_index].append(
                messagedata._rasterize_ink(ink.reshape(shape))
            )
            counts[class_index] += 1

        missing = [class_names[i] for i, count in enumerate(counts) if count == 0]
        if missing:
            raise ValueError("No training examples for classes: " + ", ".join(missing))

        features = np.concatenate([np.stack(group) for group in examples if group])
        labels = np.concatenate(
            [
                np.full(count, class_names[index])
                for index, count in enumerate(counts)
                if count
            ]
        )
        return features, labels

    @staticmethod
    def _rasterize_ink(ink):
        """Convert delta-encoded QuickDraw strokes to a 28-by-28 bitmap."""
        points = np.cumsum(ink[:, :2], axis=0)
        minimum = points.min(axis=0)
        maximum = points.max(axis=0)
        span = maximum - minimum
        scale = 24 / max(float(span.max()), 1e-6)
        points = (points - minimum) * scale + 2
        points = np.rint(points).astype(int)

        image = np.zeros((28, 28), dtype=np.float32)
        for index, (x, y) in enumerate(points):
            image[y, x] = 1.0
            if index == 0 or ink[index - 1, 2] >= 1:
                continue

            previous_x, previous_y = points[index - 1]
            steps = max(abs(x - previous_x), abs(y - previous_y), 1)
            for step in range(steps + 1):
                line_x = round(previous_x + (x - previous_x) * step / steps)
                line_y = round(previous_y + (y - previous_y) * step / steps)
                image[line_y, line_x] = 1.0
        return image.reshape(BITMAP)

    @staticmethod
    def _split(features, labels):
        classes, counts = np.unique(labels, return_counts=True)
        if len(classes) < 2 or np.any(counts < 3):
            raise ValueError("Each class needs at least three examples for train/dev/test")

        train_features, dev_features, test_features = [], [], []
        train_labels, dev_labels, test_labels = [], [], []
        for class_name, count in zip(classes, counts):
            class_features = features[labels == class_name]
            class_train, remaining = train_test_split(
                class_features,
                test_size=max(2, int(np.ceil(count * 0.2))),
                random_state=RANDOM_SEED,
            )
            class_dev, class_test = train_test_split(
                remaining,
                test_size=0.5,
                random_state=RANDOM_SEED,
            )
            train_features.append(class_train)
            dev_features.append(class_dev)
            test_features.append(class_test)
            train_labels.append(np.full(len(class_train), class_name))
            dev_labels.append(np.full(len(class_dev), class_name))
            test_labels.append(np.full(len(class_test), class_name))

        return (
            np.concatenate(train_features),
            np.concatenate(dev_features),
            np.concatenate(test_features),
            np.concatenate(train_labels),
            np.concatenate(dev_labels),
            np.concatenate(test_labels),
        )
