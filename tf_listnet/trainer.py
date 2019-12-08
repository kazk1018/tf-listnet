import argparse
import numpy as np
import tensorflow as tf
from sklearn.metrics import average_precision_score

keras = tf.keras

from tf_listnet.models import ListNet

FEATURE_LENGTH = 46


def _parse_function(example_proto):
    context_features = {
        "qid": tf.io.FixedLenFeature([], dtype=tf.int64),
    }
    sequence_features = {
        "relevance": tf.io.FixedLenSequenceFeature([], dtype=tf.float32),
        "feature": tf.io.FixedLenSequenceFeature([FEATURE_LENGTH], dtype=tf.float32),
    }
    context, sequence = tf.io.parse_single_sequence_example(
        example_proto,
        context_features=context_features,
        sequence_features=sequence_features,
    )
    return context, sequence["feature"], tf.reshape(sequence["relevance"], [-1, 1])


def permutation_top_one_probability_loss(model, x, y):
    logits = model(x)
    y_softmax = tf.nn.softmax(y, axis=1)
    return tf.nn.softmax_cross_entropy_with_logits(y_softmax, logits, axis=1)


def grad(model, x, y):
    with tf.GradientTape() as tape:
        loss_value = permutation_top_one_probability_loss(model, x, y)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


def train(train_filename, validation_filename=None, epochs=10, batch_size=8):
    train_dataset = (
        tf.data.TFRecordDataset(train_filename)
        .map(_parse_function)
        .shuffle(batch_size * 10)
        .batch(batch_size)
    )
    if validation_filename:
        validation_dataset = tf.data.TFRecordDataset(validation_filename).map(
            _parse_function
        )

    model = ListNet()

    optimizer = keras.optimizers.Adam()
    epoch_metrics = {
        "train_loss": tf.metrics.Mean(),
        "validation_mAP": tf.metrics.Mean(),
    }

    for epoch in range(epochs):

        for metric in epoch_metrics.values():
            metric.reset_states()

        for _, x, y in train_dataset:
            loss_value, grads = grad(model, x, y)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            epoch_metrics["train_loss"].update_state(loss_value)

        if validation_filename:
            scores = []
            for _, x, y in validation_dataset:
                pred = tf.nn.softmax(model(x), axis=0).numpy()
                label = tf.where(y > 0, 1, 0).numpy()

                # FIXME: is it correct to evaluate mAP?
                if not np.all(label == 0):
                    score = average_precision_score(label, pred)
                    epoch_metrics["validation_mAP"].update_state(score)

        results = {
            name: metric.result().numpy() for name, metric in epoch_metrics.items()
        }
        print("Epoch: {}, result: {}".format(epoch, results))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train",
        type=str,
        default="data/train.tfrecord",
        dest="train_filename",
        help="input tfrecord file for train",
    )
    parser.add_argument(
        "--validation",
        type=str,
        default=None,
        dest="validation_filename",
        help="input tfrecord file for validation",
    )
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument(
        "--batch_size", type=int, default=8, help="number of batch size"
    )
    args = parser.parse_args()

    train(
        args.train_filename,
        args.validation_filename,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
