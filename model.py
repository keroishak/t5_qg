import tensorflow as tf
from transformers import TFT5ForConditionalGeneration


class T5QGModel(tf.keras.Model):
    def __init__(self, model_dir, tokenizer_size, **kwargs):
        super(T5QGModel, self).__init__(**kwargs)
        input_ids = tf.keras.Input(
            shape=(None, ), dtype=tf.int32, name='input_ids')
        labels = tf.keras.Input(shape=(None, ), dtype=tf.int32, name='labels')
        decoder_attention_mask = tf.keras.Input(
            shape=(None, ), dtype=tf.int32, name='decoder_attention_mask')
        is_training = tf.keras.Input(batch_shape=(
            1,), dtype=tf.bool, name='is_training')

        self.t5_model = TFT5ForConditionalGeneration.from_pretrained(
            model_dir, from_pt=False)
        self.t5_model.resize_token_embeddings(tokenizer_size)
    
        encoder_attention_mask = tf.not_equal(input_ids, 0)
        # decoder_attention_mask = tf.not_equal(decoder_ids, 0)
        encoder_attention_mask = tf.cast(
            encoder_attention_mask, dtype=tf.int32)
        # decoder_attention_mask = tf.cast(
        #     decoder_attention_mask, dtype=tf.int32)

        input_dict = {
            'input_ids': input_ids,
            'attention_mask': encoder_attention_mask,
            'decoder_attention_mask': decoder_attention_mask,
            'labels': labels
        }
        outputs = self.t5_model(input_dict, training=is_training)
        outputs = tf.nn.softmax(outputs[1])
        self.model = tf.keras.Model(inputs=[input_ids, decoder_attention_mask, labels, is_training],
                                    outputs=outputs)
