import tensorflow as tf
from model import T5QGModel
from transformers import T5Tokenizer
import argparse
import json


def predict(qg_config_path):
    with open(qg_config_path, 'r') as f:
        config = json.load(f)

    sentencepiece_mode_path = config['sentencepiece_model_path']
    tokenizer = T5Tokenizer.from_pretrained(sentencepiece_mode_path)
    seperator_tensor = tokenizer.encode('</s>', return_tensors="tf")

    model_dir = config['t5_model_dir']
    qg = T5QGModel(model_dir)

    latest = tf.train.latest_checkpoint('checkpoint')
    qg.model.load_weights(latest).expect_partial()
    print("")
    while(True):
        print("Text: ")
        text = input()
        print("Answer: ")
        answer = input()
        input_text = 'qg_generation: ' + text
        paragraph_ids = tokenizer.encode(input_text, return_tensors="tf")
        answer_ids = tokenizer.encode(answer, return_tensors="tf")

        paragraph_ids_shape = tf.shape(paragraph_ids)
        answer_ids_shape = tf.shape(answer_ids)

        if paragraph_ids_shape[1] > 512:
            max_len = 512 - 2 - answer_ids_shape[1]
            paragraph_ids = paragraph_ids[:, :max_len]

        input_ids = tf.concat(
            [paragraph_ids, seperator_tensor, answer_ids, seperator_tensor], 1)

        result_ids = qg.t5_model.generate(input_ids)
        print(result_ids[0])
        result_text = tokenizer.decode(result_ids[0])
        print(result_text)
        print("")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str,
                        default='./config.json', help='QG configuration file')
    args = parser.parse_args()
    predict(args.config_file)
