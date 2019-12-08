import argparse
import tensorflow as tf
from collections import defaultdict


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def make_feature(column):
    _, value = column.split(":")
    return float(value)


def line_parse(line):
    data = line.strip().split()[:48]  # rel + qId + feature
    relevance = int(data[0])
    qid = int(data[1].split(":")[1])
    feature = [make_feature(column) for column in data[2:]]
    return relevance, qid, feature


def make_sequence_record(qid, relevances, features):
    context_feature = {"qid": _int64_feature(qid)}

    feature_list = dict()
    feature_list["relevance"] = tf.train.FeatureList(
        feature=[_float_feature(x) for x in relevances]
    )
    feature_list["feature"] = tf.train.FeatureList(
        feature=[_float_list_feature(feature) for feature in features]
    )

    context = tf.train.Features(feature=context_feature)
    feature_lists = tf.train.FeatureLists(feature_list=feature_list)
    example = tf.train.SequenceExample(context=context, feature_lists=feature_lists)
    return example.SerializeToString()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="data/MQ2008/Fold1/train.txt",
        help="input file (ex. I1.txt, train.txt)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/train.tfrecord",
        help="output file (ex. output.tfrecord)",
    )
    parser.add_argument(
        "--padding", action="store_true", default=False, help="padding or not"
    )

    args = parser.parse_args()

    current_qid = None

    relevances, features = defaultdict(list), defaultdict(list)
    with open(args.input) as f_in:
        for line in f_in:
            relevance, qid, feature = line_parse(line)
            relevances[qid].append(relevance)
            features[qid].append(feature)

    max_length = max([len(value) for value in relevances.values()])

    with tf.io.TFRecordWriter(args.output) as writer:
        for key in relevances.keys():

            if args.padding:
                list_size = len(relevances[key])
                if list_size < max_length:
                    for i in range(max_length - list_size):
                        relevances[key].append(0)
                        features[key].append([0.0 for i in range(46)])

            example = make_sequence_record(qid, relevances[key], features[key])
            writer.write(example)


if __name__ == "__main__":
    main()
