import tensorflow as tf


def cnn_model_fn(features, labels, mode):
    """
    Model function for CNN.
    Inspired by AlexNet + dropout in all layers + less dropouts
    Test set accuracy: 0.88
    code: evaluate_model(NN4_less_units_3, augmentation=flip_horizontally, batch_size=512, num_steps=31600)

    :param features:
    :param labels:
    :param mode:
    :return model:
    """
    curr = features["x"]

    layer_specs = [
        {"type": "conv", "kernel": [3, 3], "filters": 96, "padding": "same"},
        {"type": "relu"},
        {"type": "pool", "kernel": [3, 3], "stride": 2},
        {"type": "layer_norm"},
        {"type": "dropout", "rate": 0.5},
        {"type": "conv", "kernel": [5, 5], "filters": 256, "padding": "same"},
        {"type": "relu"},
        {"type": "pool", "kernel": [2, 2], "stride": 2},
        {"type": "layer_norm"},
        {"type": "dropout", "rate": 0.5},
        {"type": "conv", "kernel": [3, 3], "filters": 384, "padding": "same"},
        {"type": "relu"},
        {"type": "dropout", "rate": 0.5},
        {"type": "conv", "kernel": [3, 3], "filters": 384, "padding": "same"},
        {"type": "relu"},
        {"type": "dropout", "rate": 0.5},
        {"type": "pool", "kernel": [2, 2], "stride": 2},
        {"type": "conv", "kernel": [3, 3], "filters": 384, "padding": "same"},
        {"type": "relu"},
        {"type": "dropout", "rate": 0.5},

        {"type": "flatten"},
        {"type": "linear", "neurons": 1024},
        {"type": "relu"},
        {"type": "dropout", "rate": 0.5},
        {"type": "linear", "neurons": 1024},
        {"type": "relu"},
        {"type": "dropout", "rate": 0.5},
    ]

    curr = features["x"]

    dense_layers = []
    conv_layers = []


    for layer_spec in layer_specs:
        if layer_spec["type"] == "conv":
            curr = tf.layers.conv2d(
                inputs=curr,
                filters=layer_spec["filters"],
                kernel_size=layer_spec["kernel"],
                padding=layer_spec["padding"],
            )
            conv_layers.append(curr)
        elif layer_spec["type"] == "relu":
            curr = tf.nn.relu(curr)
        elif layer_spec["type"] == "pool":
            curr = tf.layers.max_pooling2d(
                inputs=curr,
                pool_size=layer_spec["kernel"],
                strides=layer_spec["stride"]
            )
        elif layer_spec["type"] == "dropout":
            curr = tf.layers.dropout(
                inputs=curr,
                rate=layer_spec["rate"],
                training=mode == tf.estimator.ModeKeys.TRAIN
            )
        elif layer_spec["type"] == "flatten":
            curr = tf.contrib.layers.flatten(curr)
        elif layer_spec["type"] == "linear":
            curr = tf.layers.dense(
                inputs=curr,
                units=layer_spec["neurons"]
            )
            dense_layers.append(curr)
        elif layer_spec["type"] == "layer_norm":
            curr = tf.contrib.layers.layer_norm(curr)
        else:
            raise Exception("Not a valid layer type")



    logits = tf.layers.dense(
        inputs=curr,
        units=10
    )

    predictions = {
        # Predictions
        "classes": tf.argmax(input=logits, axis=1),
        # Softmax Layer
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor"),
        # Unnormalized predictions
        "logits": logits
    }

    # Prediction mode for Tensorflow Custom Estimator
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Computing histograms... Only run on the final submission. Makes iterations much slower.
    tf.summary.histogram("last_fully_connected", tf.gradients(loss, dense_layers[-1]))
    tf.summary.histogram("conv1", tf.gradients(loss, conv_layers[0]))
    tf.summary.histogram("conv2", tf.gradients(loss, conv_layers[1]))
    tf.summary.histogram("conv3", tf.gradients(loss, conv_layers[2]))
    tf.summary.histogram("conv4", tf.gradients(loss, conv_layers[3]))
    tf.summary.histogram("conv5", tf.gradients(loss, conv_layers[4]))

    # Naming loss for logging purposes
    tf.identity(loss, name="loss")

    # Computing minibatch accuracy
    minibatch_accuracy = tf.metrics.accuracy(
        labels=labels, predictions=predictions["classes"])

    # Training mode for Tensorflow Custom Estimator
    if mode == tf.estimator.ModeKeys.TRAIN:
        # Renaming accuracy for logging purposes
        tf.identity(minibatch_accuracy[1], name="train_accuracy")
        tf.summary.scalar("accuracy", minibatch_accuracy[1])

        global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = 1e-4
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                   2000, 0.96, staircase=True)

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-4)
        train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Evaluation mode for Tensorflow Custom Estimator
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops={"accuracy": minibatch_accuracy}
    )
