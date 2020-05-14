import tensorflow as tf
from transformers import *
from transformers import BertTokenizer, TFBertForSequenceClassification, glue_convert_examples_to_features
from transformers.configuration_bert import BertConfig
import json
# based on tutorial: https://towardsdatascience.com/best-practices-for-nlp-classification-in-tensorflow-2-0-a5a3d43b7b73
# modified to optimize for the dataset from project #1: dataset #2

BATCH_SIZE = 32
EVAL_BATCH_SIZE = BATCH_SIZE * 2

tr_ds = tf.data.TFRecordDataset("data/train.tfrecord")
val_ds = tf.data.TFRecordDataset("data/validate.tfrecord")
test_ds = tf.data.TFRecordDataset("data/test.tfrecord")

feature_spec = {
    'idx': tf.io.FixedLenFeature([], tf.int64),
    'sentence': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.int64)
}

def parse_example(example_proto):
  return tf.io.parse_single_example(example_proto, feature_spec)

def clean_string(features):
    revised_sentence = tf.strings.regex_replace(features['sentence'], "\(R\)", "", replace_global=True)
    revised_sentence = tf.strings.regex_replace(revised_sentence, "\(L\)'", "", replace_global=True)
    revised_sentence = tf.strings.regex_replace(revised_sentence, '(January|February|Mar|April|May|June|July|August|September|October|November|December)\s\d{2}\s\d{4}', ' ')
    revised_sentence = tf.strings.regex_replace(revised_sentence, "\\n", "", replace_global=True)
    features['sentence'] = revised_sentence
    print(revised_sentence)
    return features

tr_parse_ds = tr_ds.map(parse_example)
val_parse_ds = val_ds.map(parse_example)
test_parse_ds = test_ds.map(parse_example)

# tr_clean_ds = tr_parse_ds.map(lambda features: clean_string(features))
# val_clean_ds = val_parse_ds.map(lambda features: clean_string(features))
# test_clean_ds = toptimize for est_parse_ds.map(lambda features: clean_string(features))
tr_clean_ds = tr_parse_ds
val_clean_ds = val_parse_ds
test_clean_ds = test_parse_ds

with open('data/info.json') as json_file:
    data_info = json.load(json_file)
train_examples = data_info['train_length']
valid_examples = data_info['validation_length']
test_examples = data_info['test_length']

USE_XLA = False
USE_AMP = False
tf.config.optimizer.set_jit(USE_XLA)
tf.config.optimizer.set_experimental_options({"auto_mixed_precision": USE_AMP})

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
config = BertConfig("bert_config.json")
model = TFBertForSequenceClassification.from_pretrained('bert-base-cased', config=config)

#Training Dataset
train_dataset = glue_convert_examples_to_features(examples=tr_clean_ds, tokenizer=tokenizer
                                                  , max_length=128, task='sst-2'
                                                  , label_list =['1', '3'])
train_dataset = train_dataset.shuffle(train_examples).batch(BATCH_SIZE).repeat(-1)

#Validation Dataset
valid_dataset = glue_convert_examples_to_features(examples=val_clean_ds, tokenizer=tokenizer
                                                  , max_length=128, task='sst-2'
                                                  , label_list =['1', '3'])
valid_dataset = valid_dataset.batch(EVAL_BATCH_SIZE)

opt = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
model.compile(optimizer=opt, loss=loss, metrics=[metric])

train_steps = train_examples//BATCH_SIZE *2
valid_steps = valid_examples//EVAL_BATCH_SIZE *2
model.summary()
history = model.fit(train_dataset, epochs=1, steps_per_epoch=train_steps,
                    validation_data=valid_dataset, validation_steps=valid_steps)

test_dataset = glue_convert_examples_to_features(examples=test_clean_ds, tokenizer=tokenizer
                                                  , max_length=128, task='sst-2'
                                                  , label_list =['1', '3'])
test_dataset = test_dataset.batch(EVAL_BATCH_SIZE)

print("Evaluating the results of the model - test dataset")
print(model.evaluate(test_dataset))
