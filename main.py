import tensorflow as tf
import data_utils
import numpy as np
import os
import math
from time import time
import sys

import NN4_less_units_3


tf.logging.set_verbosity(tf.logging.INFO)

MODELS_DIR = "./Models"
OUTPUTS_DIR = "./Outputs"


def identity(x, y):
    """
    Identity transformation for data augmentation
    :param x: features, shape [n_images, width, height, channels]
    :param y: target class
    :return: identity
    """
    return x, y


def random_crop(X, y, size=24, n_random_per_image=4):
    original_size = X.shape[1]

    def crop(Z):
        max_displace = original_size-size
        displace_x = np.random.randint(0, max_displace+1)
        displace_y = np.random.randint(0, max_displace + 1)
        X2 = Z.copy()
        X2[:displace_x, ...] = 0
        X2[displace_x+size:, ...] = 0
        X2[:, :displace_y, :] = 0
        X2[:, displace_y + size:, :] = 0
        return X2
    imgs = []
    new_y = []
    for idx in range(X.shape[0]):
        for j in range(n_random_per_image):
            imgs.append(crop(X[idx, ...]))
            new_y.append(y[idx])
    imgs = np.array(imgs)
    new_y = np.array(new_y)
    return imgs, new_y

def flip_horizontally(X, y):
    """
    Flip images horizontally for data augmentation
    :param X:
    :param y:
    :return: original images plus images flipped
    """
    train_X_flip = []

    for i in range(X.shape[0]):
        train_X_flip.append(np.fliplr(X[i]))

    train_X_flip = np.array(train_X_flip)

    X = np.vstack([X, train_X_flip])
    y = np.tile(y, 2)
    return X, y


def all_augment(X, y):
    """
    Runs all augmentations
    :param X:
    :param y:
    :return X_augmented, y_augmented:
    """
    X, y = flip_horizontally(X, y)
    X, y = random_crop(X, y)
    return X, y


def evaluate_model(module, augmentation=identity, batch_size=256, num_steps=20000):
    """
    Evaluate the model generated by cnn_model_fn
    :param module: module containing cnn_model_fn
    :param augmentation: augmentation function. Default: identity
    :param batch_size: batch size
    :param num_steps: number of steps for optimization
    :return: None
    """

    model_name = module.__name__

    print("Evaluating model " + model_name)

    cnn_model_fn = module.cnn_model_fn
    os.makedirs(MODELS_DIR, exist_ok=True)

    os.makedirs(os.path.join(OUTPUTS_DIR, model_name), exist_ok=True)

    print("Loading data files...")

    train_X, train_y, val_X, val_y = data_utils.get_train_validation_dataset()
    test_X_sorted_number = data_utils.get_test_dataset_sorted_number()
    test_X_sorted_name = data_utils.get_test_dataset_sorted_name()

    print("Augmenting data...")
    train_X, train_y = augmentation(train_X, train_y)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_X},
        y=train_y,
        batch_size=batch_size,
        num_epochs=None,
        shuffle=True
    )

    val_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": val_X},
        y=val_y,
        batch_size=50,
        num_epochs=1,
        shuffle=False
    )

    test_sorted_number_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": test_X_sorted_number},
        batch_size=50,
        num_epochs=1,
        shuffle=False
    )

    test_sorted_name_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": test_X_sorted_name},
        batch_size=50,
        num_epochs=1,
        shuffle=False
    )

    classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn,
        model_dir=os.path.join(MODELS_DIR, model_name),
    )

    print("Training neural network...")

    # tensors_to_log = {"loss": "loss", "accuracy": "accuracy"}
    tensors_to_log = {"loss": "loss", "train_accuracy": "train_accuracy"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    steps_before_eval = 400

    time_start = time()

    for i in range(int(math.ceil(num_steps/(1.0*steps_before_eval)))):
        classifier.train(input_fn=train_input_fn, hooks=[logging_hook], steps=steps_before_eval)
        val_result = classifier.evaluate(
            input_fn=val_input_fn
        )
        sys.stderr.write('\n Test set accuracy: {accuracy:0.3f}\n'.format(**val_result))

    time_end = time()

    print("Computing accuracy...")

    val_result = classifier.evaluate(
        input_fn=val_input_fn
    )

    print('\n Test set accuracy: {accuracy:0.3f}\n'.format(**val_result))

    with open(os.path.join(OUTPUTS_DIR, model_name, "trainingInfo.txt"), 'w') as f:
        f.write(('"'*50) + "\n")
        f.write('Model: {}\n'.format(model_name))
        f.write(('"' * 50) + "\n")
        f.write('Test set accuracy: {accuracy:0.3f}\n'.format(**val_result))
        f.write('Training Time: {:.3f}s\n'.format(time_end - time_start))
        f.write('Augmentation: {}\n'.format(str(augmentation)))
        f.write('Training data size: {}\n'.format(train_X.shape[0]))
        f.write('Validation data size: {}\n'.format(val_X.shape[0]))
        f.write('Batch Size: {}\n'.format(batch_size))
        f.write('Num steps: {}\n'.format(num_steps))

    print("Saving predictions")

    predictions = classifier.predict(
        input_fn=test_sorted_number_input_fn
    )

    data_utils.save_preds_csv(os.path.join(OUTPUTS_DIR, model_name, "predictions.csv"), predictions)

    predictionsNpy = classifier.predict(
        input_fn=test_sorted_name_input_fn
    )

    data_utils.save_preds_npy(os.path.join(OUTPUTS_DIR, model_name, "predictions.npy"),
                              pred_shape=[10, test_X_sorted_name.shape[0]], pred_list=predictionsNpy)

# 0.88 accuracy
# evaluate_model(NN4_less_units_3, augmentation=flip_horizontally, batch_size=512, num_steps=31600)


evaluate_model(NN4_less_units_3, augmentation=flip_horizontally, batch_size=512, num_steps=31600)