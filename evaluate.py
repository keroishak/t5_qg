import tensorflow as tf
from model import T5QGModel
from transformers import T5Tokenizer
import argparse
import json
import csv
from bleu import compute_bleu


def evaluate(qg_config_path, data_file):
    with open(qg_config_path, 'r') as f:
        config = json.load(f)

    sentencepiece_mode_path = config['sentencepiece_model_path']
    tokenizer = T5Tokenizer.from_pretrained(sentencepiece_mode_path)
    eos_id = tokenizer.encode(tokenizer.eos_token, return_tensors="tf")
    model_dir = config['t5_model_dir']
    qg = T5QGModel(model_dir)

    latest = tf.train.latest_checkpoint('checkpoint')
    qg.model.load_weights(latest).expect_partial()

    data_file_without_extension = data_file.split('.')[0]
    csv_file = open(data_file_without_extension + '_result.csv', mode='w')
    fieldnames = ['paragraph', 'answer/query',
                  'golden question', 'generated question']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    dataset = tf.data.TFRecordDataset([data_file])
    golden_reference_q = []
    generated_q = []
    for record in dataset:
        example = tf.train.Example()
        example.ParseFromString(record.numpy())
        paragraph = example.features.feature['paragraph'].bytes_list.value[
            0].decode('utf-8')
        answer = example.features.feature['answer'].bytes_list.value[0].decode(
            'utf-8')
        golden_question = example.features.feature['question'].bytes_list.value[
            0].decode('utf-8')

        paragraph_text = 'qg_generation: ' + paragraph
        paragraph_ids = tokenizer.encode(paragraph_text, return_tensors="tf")
        answer_ids = tokenizer.encode(answer, return_tensors="tf")

        paragraph_ids_shape = tf.shape(paragraph_ids)
        answer_ids_shape = tf.shape(answer_ids)

        if paragraph_ids_shape[1] + answer_ids_shape[1] + 2 > 512:
            max_len = 512 - 2 - answer_ids_shape[1]
            paragraph_ids = paragraph_ids[:, :max_len]
        input_ids = tf.concat([paragraph_ids, eos_id, answer_ids, eos_id], 1)
        result_ids = qg.t5_model.generate(input_ids)
        generated_question = tokenizer.decode(result_ids[0])
        writer.writerow({'paragraph': paragraph_text, 'answer/query': answer,
                         'golden question': golden_question, 'generated question': generated_question})

        golden_reference_q.append(golden_question.split(' '))
        generated_q.append(generated_question.split(' '))

    csv_file.close()
    bleu, precisions, bp, ratio, question_length, reference_length = compute_bleu(
        golden_reference_q, generated_q, max_order=4)
    print("BLEU: %f" % (bleu))
    print("Precisions: ", precisions)
    print("Brevity Penalty: %f" % (bp))
    print("Ratio: %f" % (ratio))
    print("Question length: %f" % (question_length))
    print("Reference length: %f" % (reference_length))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str,
                        default='./config.json', help='QG configuration file')
    parser.add_argument('--data_file', type=str, required=True,
                        help='File path of type tfrecord')
    args = parser.parse_args()
    evaluate(args.config_file, args.data_file)
