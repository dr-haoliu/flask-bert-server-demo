from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from flask import Flask, render_template
import os
from flask import request, jsonify
import json
import pickle
from datetime import datetime
import tensorflow as tf
from tensorflow import keras as K
import numpy as np
import codecs

import sys
sys.path.append('../..')

from bert_base.run_classifier_ner_chia_3 import create_model_2, InputFeatures
from bert_base import tokenization, modeling


def get_labels(data_dir):
    def read_label_from_file(data_dir, filename, task_name='ner'):
        tags = set()
        with codecs.open(os.path.join(data_dir, filename), mode="r", encoding="utf-8") as f:
            for line in f:
                line = line.lstrip().rstrip()
                if len(line) == 0 or line.startswith("-DOCSTART-"):  # means read whole one sentence
                    continue
                else:
                    word, ner = line.split("\t")
                    if task_name == "ner":
                        tag = ner
                    tags.add(tag)
        return tags

    tags = read_label_from_file(data_dir, "train.txt")
    labels = tags.union(set(["APAD", "[CLS]", "[SEP]"]))
    labels = list(labels)
    print(labels)
    labels.sort()
    print(labels)
    return labels



# model_dir = r'../../output'
model_dir = 'D:/pycharm_projects/bert/OUTPUT/chia'
bert_dir = 'D:/pycharm_projects/bert/google_bert_models/cased_L-12_H-768_A-12'
data_dir = 'D:/pycharm_projects/bert/s7/processed_chia'

num_labels =18

label_list = get_labels(data_dir)

is_training=False
use_one_hot_embeddings=False
batch_size=1
max_seq_length = 128

gpu_config = tf.ConfigProto()
gpu_config.gpu_options.allow_growth = True
sess=tf.Session(config=gpu_config)
model=None

global graph
input_ids_p, input_mask_p, label_ids_p, segment_ids_p = None, None, None, None

print('checkpoint path:{}'.format(os.path.join(model_dir, "checkpoint")))
if not os.path.exists(os.path.join(model_dir, "checkpoint")):
    raise Exception("failed to get checkpoint. going to return ")


graph = tf.get_default_graph()
with graph.as_default():
    print("going to restore checkpoint")
    #sess.run(tf.global_variables_initializer())
    input_ids_p = tf.placeholder(tf.int32, [batch_size, max_seq_length], name="input_ids")
    input_mask_p = tf.placeholder(tf.int32, [batch_size, max_seq_length], name="input_mask")

    bert_config = modeling.BertConfig.from_json_file(os.path.join(bert_dir, 'bert_config.json'))
    (total_loss, logits, pred_ids) = create_model_2(
        bert_config=bert_config, is_training=False, input_ids=input_ids_p, input_mask=input_mask_p, segment_ids=None,
        label_ids=None, num_labels=num_labels, use_one_hot_embeddings=False)

    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(model_dir))

tokenizer = tokenization.FullTokenizer(
        vocab_file=os.path.join(bert_dir, 'vocab.txt'), do_lower_case=True)


app = Flask(__name__)

@app.route('/ner_predict_service', methods=['GET'])
def ner_predict_service():
    """
    do online prediction. each time make prediction for one instance.
    you can change to a batch if you want.
    :param line: a list. element is: [dummy_label,text_a,text_b]
    :return:
    """
    def convert(line):
        feature = convert_single_example(0, line, label_list, max_seq_length, tokenizer)
        input_ids = np.reshape([feature.input_ids],(batch_size, max_seq_length))
        input_mask = np.reshape([feature.input_mask],(batch_size, max_seq_length))
        segment_ids = np.reshape([feature.segment_ids],(batch_size, max_seq_length))
        label_ids =np.reshape([feature.label_ids],(batch_size, max_seq_length))
        return input_ids, input_mask, segment_ids, label_ids

    global graph
    with graph.as_default():
        result = {}
        result['code'] = 0
        try:
            sentence = request.args['query']
            result['query'] = sentence
            start = datetime.now()
            if len(sentence) < 2:
                print(sentence)
                result['data'] = ['O'] * len(sentence)
                return json.dumps(result)
            # sentence = tokenizer.tokenize(sentence)
            # print('your input is:{}'.format(sentence))
            input_ids, input_mask, segment_ids, label_ids = convert(sentence)


            feed_dict = {input_ids_p: input_ids,
                         input_mask_p: input_mask}
            # run session get current feed_dict result
            pred_ids_result = sess.run([pred_ids], feed_dict)
            # pred_ids_result = pred_ids_result[0][0]
            pred_label_result = convert_id_to_label(pred_ids_result, label_list)
            print(pred_label_result)
            #todo: 组合策略
            result['input_tokens'] = convert_input_id_to_tokens(input_ids)
            result['data'] = pred_label_result
            print('time used: {} sec'.format((datetime.now() - start).total_seconds()))
            return json.dumps(result)
        except:
            result['code'] = -1
            result['data'] = 'error'
            return json.dumps(result)


def convert_input_id_to_tokens(input_ids):
    remove_list = ['[CLS]', '[PAD]', '[SEP]']
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    result =[]
    for token in tokens:
        if token in remove_list:
            continue
        else:
            result.append(token)
    return result

def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    input_tokens = example.split(" ")
    token_labels = ['O'] * len(input_tokens)

    pre_tokens = []
    pre_labels = []

    for i, word in enumerate(input_tokens):
        # word tokenize, if not in vocab.txt of bert, it will use WordPiece. For word being tokenized, add label 'X'
        token = tokenizer.tokenize(word)
        pre_tokens.extend(token)
        pre_label = token_labels[i]
        for m in range(len(token)):
            pre_labels.append(pre_label)
            # if m == 0:
            #     pre_labels.append(pre_label)
            # else:
            #     pre_labels.append(pre_label)

    assert len(pre_tokens) == len(pre_labels), "{} \t {}".format(pre_tokens, pre_labels)

    # Account for [CLS] and [SEP] with "- 2"
    if len(pre_tokens) > max_seq_length - 2:
        pre_tokens = pre_tokens[0:(max_seq_length - 2)]
        pre_labels = pre_labels[0:(max_seq_length - 2)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    segment_ids = []
    label_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    label_ids.append(label_map["[CLS]"])

    for i, token in enumerate(pre_tokens):
        tokens.append(token)
        segment_ids.append(0)
        label_ids.append(label_map[pre_labels[i]])

    tokens.append("[SEP]")
    segment_ids.append(0)
    label_ids.append(label_map["[SEP]"])

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        label_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length

    # label_id = label_map[example.label]

    if ex_index < 5:
        tf.logging.info("*** Example ***")
        # tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label: %s " % " ".join([str(x) for x in label_ids]))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
        is_real_example=True)
    return feature


def convert_id_to_label(pred_ids_result, label_list):
    result = []
    for row in range(batch_size):
        curr_seq = []
        for ids in pred_ids_result[row][0]:
            if ids == 0:
                break
            curr_label = label_list[ids]
            if curr_label in ['[CLS]', '[SEP]']:
                continue
            curr_seq.append(curr_label)
        result.append(curr_seq)
    return result


@app.route('/')
def hello_world():
    return render_template('index.html')  #'Hello World!'


if __name__ == '__main__':
    app.run(debug=True)
