import tensorflow as tf
import numpy as np
from transformers import T5TokenizerFast
import enum


class DatasetPipeline():
    def __init__(self, dataset_file, batch_size, sentencepiece_file_path, process_mode, slim_dataset=False):
        self.feature_description = {
            'input_ids': tf.io.FixedLenFeature([512], tf.int64),
            'label_ids': tf.io.FixedLenFeature([60], tf.int64),
            'decoder_attention_mask': tf.io.FixedLenFeature([60], tf.int64)
        }
        dataset = tf.data.TFRecordDataset([dataset_file])
        # self.start_answer_special_token = "<ANS>"
        # self.end_answer_special_token = "</ANS>"
        self.tokenizer = T5TokenizerFast.from_pretrained(sentencepiece_file_path)
        # special_tokens_dict = {'additional_special_tokens': [self.start_answer_special_token, self.end_answer_special_token]}
        # self.tokenizer.add_special_tokens(special_tokens_dict)

        # dataset = dataset.map(self.__tf_tokenize).filter(self.filter_empty_record).padded_batch(batch_size, padded_shapes=([-1], [-1], [-1])).map(self.labels_padding).map(
        #     lambda inp_ids, target, decoder_mask: ({'input_ids': inp_ids, 'decoder_attention_mask': decoder_mask}, target))
        if process_mode == ProcessMode.Training:
            dataset = dataset.map(self.decode_record).batch(batch_size).map(
                lambda record: ({'input_ids': record['input_ids'], 'decoder_attention_mask': record['decoder_attention_mask'], 'labels': record['label_ids'], 'is_training': True}, record['label_ids']))
        elif process_mode == ProcessMode.Validation:
            dataset = dataset.map(self.decode_record).batch(batch_size).map(
                lambda record: ({'input_ids': record['input_ids'], 'decoder_attention_mask': record['decoder_attention_mask'], 'labels': record['label_ids'], 'is_training': False}, record['label_ids']))             
        else:
            dataset = dataset.map(self.decode_record).batch(batch_size)
                    
        if slim_dataset:
            dataset = dataset.take(2000)

        self.dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    
    def decode_record(self, record):
        return tf.io.parse_single_example(record, self.feature_description)

    def get_tokenizer_size(self):
        return len(self.tokenizer)

class ProcessMode(enum.Enum):
   Training = 1
   Testing = 2
   Validation = 3