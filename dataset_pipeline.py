import tensorflow as tf
import numpy as np
from transformers import T5Tokenizer


class DatasetPipeline():
    def __init__(self, dataset_file, batch_size, sentencepiece_file_path, target_max_len, is_training, slim_dataset=False):

        dataset = tf.data.TFRecordDataset([dataset_file])
        self.target_max_len = target_max_len
        self.tokenizer = T5Tokenizer.from_pretrained(sentencepiece_file_path)
        self.eos_id = self.tokenizer.encode(
            self.tokenizer.eos_token, return_tensors="tf")
        self.pad_id = self.tokenizer.encode(
            self.tokenizer.pad_token, return_tensors="tf")

        dataset = dataset.map(self.__tf_tokenize).padded_batch(batch_size, padded_shapes=([-1], [-1], [-1])).map(self.labels_padding).map(
            lambda inp_ids, target, decoder_mask: ({'input_ids': inp_ids, 'decoder_attention_mask': decoder_mask, 'labels': target, 'is_training': is_training}, target))

        if slim_dataset:
            dataset = dataset.take(5)

        self.dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    def labels_padding(self, inp_ids, target, decoder_mask):
        target_boolean_mask = tf.equal(target, 0)
        target_mask = tf.cast(target_boolean_mask, dtype=tf.int32) * -100
        target = target + target_mask
        return inp_ids, target, decoder_mask

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
            [paragraph_ids, self.eos_id, answer_ids, self.eos_id], 1)
        target = self.tokenizer.encode(question, return_tensors="tf")
        target_shape = tf.shape(target)

        if target_shape[1] > self.target_max_len:
            target = target[:, :self.target_max_len]

        target = tf.concat([target, self.eos_id], 1)

        return tf.squeeze(input_ids), tf.squeeze(target), np.ones(shape=[tf.shape(target)[1]])

    def __tf_tokenize(self, record):
        return tf.py_function(self.tokenize, [record],
                              [tf.int32, tf.int32, tf.int32])
