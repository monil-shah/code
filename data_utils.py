import scipy.misc
import numpy as np
import os

"""
Environment variables
"""


CIFAR_ROOT = os.environ["CIFAR_ROOT"]
CIFAR_TRAIN = os.environ["CIFAR_TRAIN"]
CIFAR_TEST = os.environ["CIFAR_TEST"]
CIFAR_LABELS = os.environ["CIFAR_LABELS"]


"""
Labels Management
"""

_labels_info = None


def _load_labels():
    """
    Loads the labels from file CIFAR_LABELS. Creates a global dictionary
    _labels_info with {'array':[...], 'dict':{...}}. _labels_info is used
    by functions map_idx_to_label and map_label_to_idx
    """
    global _labels_info
    with open(CIFAR_LABELS) as f:
        labels_array = [line.strip() for line in f.readlines()]
        labels_dict = {label: idx for idx, label in enumerate(labels_array)}
        _labels_info = {
            "array": labels_array,
            "dict": labels_dict
        }


def map_idx_to_label(idx):
    """
    Maps an index to a label, using _labels_info.
    :param idx: index (int)
    :return label: (string)
    """
    if type(idx) is not int:
        raise TypeError("idx should be int")
    if _labels_info is None:
        _load_labels()
    return _labels_info["array"][idx]


def map_label_to_idx(label):
    """
    Maps a label to a int, using _labels_info.
    :param label: (string)
    :return index: (int)
    """
    if type(label) is not str:
        raise TypeError("label should be str")
    if _labels_info is None:
        _load_labels()
    return _labels_info["dict"][label]



"""
Data loading
"""

training_mean = None

def get_train_validation_dataset(train_percentage=0.9):
    """
    :return: [train_X (ndarray),  train_y (list), validation_X (ndarray),  validation_y (list)]
    """
    def class_from_filename(filename):
        idx_start = filename.rfind("_") + 1
        idx_end = filename.rfind(".png")
        return map_label_to_idx(filename[idx_start:idx_end])
    global training_mean
    filenames = os.listdir(CIFAR_TRAIN)
    filenames.sort()
    filenames = filter(lambda f: f.find(".png") > -1, filenames)
    filenames = list(map(lambda f: os.path.join(CIFAR_TRAIN, f), filenames))
    imgs = [scipy.misc.imread(f) for f in filenames]
    imgs = np.array(imgs, dtype=np.float32)
    classes = np.array(list(map(class_from_filename, filenames)), dtype=np.int32)
    n = classes.shape[0]
    ids = np.random.choice(n, n, replace=False)
    n_train = int(n * train_percentage)
    ids_train = ids[:n_train]
    ids_val = ids[n_train:]
    train_X = imgs[ids_train, ...]
    train_y = classes[ids_train]
    training_mean = train_X.mean(axis=0)
    train_X = train_X - training_mean
    val_X = imgs[ids_val, ...]
    val_y = classes[ids_val]
    val_X = val_X - training_mean
    return train_X, train_y, val_X, val_y

def get_test_dataset_sorted_name():
    """
    :return: imgs (ndarray)
    """
    global training_mean
    if training_mean is None:
        a,b,c,d = get_train_validation_dataset()
    filenames = os.listdir(CIFAR_TEST)
    filenames.sort()
    filenames = filter(lambda f: f.find(".png") > -1, filenames)
    filenames = list(map(lambda f: os.path.join(CIFAR_TEST, f), filenames))
    imgs = [scipy.misc.imread(f) for f in filenames]
    imgs = np.array(imgs, dtype=np.float32)
    imgs = imgs - training_mean
    return imgs


def get_test_dataset_sorted_number():
    """
    :return: imgs (ndarray)
    """
    global training_mean
    if training_mean is None:
        a,b,c,d = get_train_validation_dataset()
    n_files = 5000
    filenames = ["0" + str(i) + ".png" for i in range(1, n_files + 1)]
    filenames = list(map(lambda f: os.path.join(CIFAR_TEST, f), filenames))
    imgs = [scipy.misc.imread(f) for f in filenames]
    imgs = np.array(imgs, dtype=np.float32)
    imgs = imgs - training_mean
    return imgs


def save_preds_csv(fname, pred_list):
    with open(fname, 'w') as f:
        f.write("id,label\n")
        for idx, pred in enumerate(pred_list):
            f.write(str(idx) + "," + map_idx_to_label(int(pred["classes"])) + "\n")

def save_preds_npy(fname, pred_shape, pred_list):
    logits = np.zeros(pred_shape)
    for idx, pred in enumerate(pred_list):
        logits[:, idx] = pred["logits"]
    np.save(fname, logits)