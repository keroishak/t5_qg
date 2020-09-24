import tensorflow as tf
from model import T5QGModel
from transformers import T5Tokenizer
import argparse
import json
import csv

def predict(qg_config_path, input_file):
    with open(qg_config_path, 'r') as f:
        config = json.load(f)

    sentencepiece_mode_path = config['sentencepiece_model_path']
    tokenizer = T5Tokenizer.from_pretrained(sentencepiece_mode_path)
    eos_id = tokenizer.encode(tokenizer.eos_token, return_tensors="tf")
    model_dir = config['t5_model_dir']
    qg = T5QGModel(model_dir)

    latest = tf.train.latest_checkpoint(config['checkpoint_path'])
    qg.model.load_weights(latest).expect_partial()

    csv_file = open('[new_QG]AP_articles.csv', mode='w')
    fieldnames = ['paragraph', 'answer/query', 'generated question']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    with open(input_file,'r') as file_reader:
        for line in file_reader:
            line_segments = line.split('</s>')
            paragraph_text = line_segments[0].strip()
            answer_text = line_segments[1].strip()
            paragraph_ids = tokenizer.encode(paragraph_text, return_tensors="tf")
            answer_ids = tokenizer.encode(answer_text, return_tensors="tf")

            paragraph_ids_shape = tf.shape(paragraph_ids)
            answer_ids_shape = tf.shape(answer_ids)

            if paragraph_ids_shape[1] + answer_ids_shape[1] + 2 > 512:
                max_len = 512 - 2 - answer_ids_shape[1]
                paragraph_ids = paragraph_ids[:, :max_len]
            input_ids = tf.concat([paragraph_ids, eos_id, answer_ids, eos_id], 1)
            result_ids = qg.t5_model.generate(input_ids)
            generated_question = tokenizer.decode(result_ids[0])
            print(generated_question)
            writer.writerow({'paragraph': paragraph_text, 'answer/query': answer_text, 'generated question': generated_question })
    csv_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str,
                        default='./config.json', help='QG configuration file')
    parser.add_argument('--file_path', type=str, help='Path to .txt file where each line represents an example')                    
    args = parser.parse_args()
    predict(args.config_file, args.file_path)
