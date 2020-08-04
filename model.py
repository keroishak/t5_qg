import tensorflow as tf
from transformers import TFT5ForConditionalGeneration


class T5QGModel(tf.keras.Model):
    def __init__(self, model_dir, **kwargs):
        super(T5QGModel, self).__init__(**kwargs)
        inp = tf.keras.Input(shape=(None, ), dtype=tf.int32, name='input')
        tar = tf.keras.Input(shape=(None, ), dtype=tf.int32, name='target')

        model = TFT5ForConditionalGeneration.from_pretrained(
            model_dir, from_pt=False)
        encoder_attention_mask = tf.not_equal(inp, 0)
        decoder_attention_mask = tf.not_equal(tar, 0)
        encoder_attention_mask = tf.cast(
            encoder_attention_mask, dtype=tf.int32)
        decoder_attention_mask = tf.cast(
            decoder_attention_mask, dtype=tf.int32)
        # tf.print(encoder_attention_mask)
        # tf.print(decoder_attention_mask)
        outputs = model(inp, decoder_input_ids=tar, attention_mask=encoder_attention_mask,
                        decoder_attention_mask=decoder_attention_mask)
        self.model = tf.keras.Model(inputs=[inp, tar],
                                    outputs=outputs[0])
                                    
