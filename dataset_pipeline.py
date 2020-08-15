import tensorflow as tf
import numpy as np
from transformers import T5Tokenizer


class DatasetPipeline():
    def __init__(self, dataset_file, batch_size, sentencepiece_file_path, target_max_len, slim_dataset=False):

        dataset = tf.data.TFRecordDataset([dataset_file])
        self.target_max_len = target_max_len
        self.tokenizer = T5Tokenizer.from_pretrained(sentencepiece_file_path)
        self.seperator_tensor = self.tokenizer.encode(
            '</s>', return_tensors="tf")

        dataset = dataset.map(self.__tf_tokenize).padded_batch(batch_size, padded_shapes=(
            [-1], [-1])).map(lambda x, y: ({'input': x, 'target': y}, y))

        if slim_dataset:
            dataset = dataset.take(5)

        self.dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    def tokenize(self, record):
        example = tf.train.Example()
        example.ParseFromString(record.numpy())
        paragraph = example.features.feature['paragraph'].bytes_list.value[
            0].decode('utf-8')
        answer = example.features.feature['answer'].bytes_list.value[0].decode(
            'utf-8')
        question = example.features.feature['question'].bytes_list.value[
            0].decode('utf-8')

        input_text = 'qg_generation: ' + paragraph
        paragraph_ids = self.tokenizer.encode(input_text, return_tensors="tf")
        answer_ids = self.tokenizer.encode(answer, return_tensors="tf")

        paragraph_ids_shape = tf.shape(paragraph_ids)
        answer_ids_shape = tf.shape(answer_ids)

        if paragraph_ids_shape[1] + answer_ids_shape[1] + 2 > 512:
            max_len = 512 - 2 - answer_ids_shape[1]
            paragraph_ids = paragraph_ids[:, :max_len]

        input_ids = tf.concat(
            [paragraph_ids, self.seperator_tensor, answer_ids, self.seperator_tensor], 1)
        target = self.tokenizer.encode(question, return_tensors="tf")
        target_shape = tf.shape(target)

        if target_shape[1] > self.target_max_len - 1:
            target = target[:, :self.target_max_len - 1]
        target = tf.concat([target, self.seperator_tensor], 1)

        return tf.squeeze(input_ids), tf.squeeze(target)

    def __tf_tokenize(self, record):
        return tf.py_function(self.tokenize, [record],
                              [tf.int32, tf.int32])
