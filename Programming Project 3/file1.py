#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import pandas as pd
from tensorflow.python.lib.io.tf_record import TFRecordWriter
import time
import json
# based on tutorial: https://towardsdatascience.com/best-practices-for-nlp-classification-in-tensorflow-2-0-a5a3d43b7b73
# modified to optimize for the dataset from project #1: dataset #2

def create_tf_example(features, label):
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'idx': tf.train.Feature(int64_list=tf.train.Int64List(value=[features[0]])),
        'sentence': tf.train.Feature(bytes_list=tf.train.BytesList(value=[features[1].encode('utf-8')])),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
    }))
    return tf_example

def convert_csv_to_tfrecord(csv, file_name):
    writer = TFRecordWriter(file_name)
    for index, row in enumerate(csv):
        features, label = row[:-1], row[-1]
        example = create_tf_example(features, label)
        writer.write(example.SerializeToString())
    writer.close()

def generate_json_info(local_file_name):
    info = {"train_length": len(train_df), "validation_length": len(validate_df),
            "test_length": len(test_df)}
    with open(local_file_name, 'w') as outfile:
        json.dump(info, outfile)

train_file = open("corpus2_train.labels", "r")
train_lines = train_file.read().splitlines()
df = pd.DataFrame(columns=['idx','utterance', 'class'])
indexNum = 1

for line in train_lines:
    split = line.split()
    sentence = open(split[0], "r")
    sentence = sentence.read()
    if split[1] == "O":
        sent = 3
    elif split[1] == "I":
        sent = 1
    df = df.append({'idx': indexNum, 'utterance': sentence, 'class': sent}, ignore_index=True)
    indexNum = indexNum+1

# with pd.option_context('display.max_colwidth', 1000):
#     display(df.head(60))

df_O = df[df['class'] == 3]
df_train_O = df_O.sample(473)
df_O_remaining = df_O.drop(df_train_O.index)
df_validate_O = df_O_remaining.sample(frac=0.5)
df_test_O = df_O_remaining.drop(df_validate_O.index)

df_I = df[df['class'] == 1]
df_train_I = df_I.sample(273)
df_I_remaining = df_I.drop(df_train_I.index)
df_validate_I = df_I_remaining.sample(frac=0.5)
df_test_I = df_I_remaining.drop(df_validate_I.index)

train_df = pd.concat([df_train_O, df_train_I]).sample(frac=1)
validate_df = pd.concat([df_validate_O, df_validate_I]).sample(frac=1)
test_df = pd.concat([df_test_O, df_test_I]).sample(frac=1)

convert_csv_to_tfrecord(train_df.values, "data/train.tfrecord")
convert_csv_to_tfrecord(validate_df.values, "data/validate.tfrecord")
convert_csv_to_tfrecord(test_df.values, "data/test.tfrecord")
generate_json_info("data/info.json")

# I then sent the DATA folder and JSON file to the server to run on the GPU
