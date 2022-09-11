from cProfile import label
import tensorflow as tf
from model import T5QGModel
from transformers import T5TokenizerFast
import argparse
import json
import csv
import numpy as np
from dataset_pipeline import DatasetPipeline, ProcessMode

def evaluate(qg_config_path, data_file, beam_search, top_k):
    with open(qg_config_path, 'r') as f:
        config = json.load(f)

    sentencepiece_mode_path = config['sentencepiece_model_path']
    tokenizer = T5TokenizerFast.from_pretrained(sentencepiece_mode_path)
    model_dir = config['t5_model_dir']
    qg = T5QGModel(model_dir, len(tokenizer))
    checkpoint_path = config['checkpoint_path']
    batch_size = config['batch_size'] + 10
    target_max_len = int(config['target_max_len'])
    # latest = tf.train.latest_checkpoint('checkpoint')
    if checkpoint_path[-1] == '/':
        checkpoint_path = checkpoint_path[:-1]
    # checkpoint_path = checkpoint_path + '/t5_qg_' + str(labels_drop_rate*100) + '_dropped.ckpt'
    
    model_checkpoint_name = checkpoint_path + '/t5_qg_without_dropping.ckpt'
    print("loading "+ model_checkpoint_name)
    qg.model.load_weights(model_checkpoint_name).expect_partial()

    data_file_without_extension = '.'.join(data_file.split('.')[:-1])
    
    csv_file = open(data_file_without_extension + '_BT_' +'t5_base_debug_7e5lr_10_epochs_24_june.csv', mode='w')

    fieldnames = ['paragraph',
                  'golden question', 'generated question - beam search', 'generated question - top k and p']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    test_input = DatasetPipeline(data_file, batch_size, sentencepiece_mode_path, ProcessMode.Testing) 

    generated_beam_array = []
    generated_top_p_array = []
    golden_questions_array = []
    paragraphs_array = []
    for record in test_input.dataset:
        input_ids = record['input_ids']
        label_ids = record['label_ids']

        if beam_search:
            result_ids = qg.t5_model.generate(input_ids, num_beams=4, max_length=target_max_len,
                length_penalty=1)
            aggregated_beam_questions = tokenizer.batch_decode(result_ids, skip_special_tokens=True)
            # aggregated_beam_questions = '\n'.join(generated_question)
            generated_beam_array+= aggregated_beam_questions

        if top_k:
            result_ids = qg.t5_model.generate(input_ids, top_p=0.9, top_k=100, do_sample=True)
            top_k_top_p_generated_question = tokenizer.batch_decode(result_ids, skip_special_tokens=True)
            generated_top_p_array+=top_k_top_p_generated_question

        paragraph = tokenizer.batch_decode(input_ids, skip_special_tokens=False)
        new_label_ids = tf.where(tf.equal(label_ids, -100), 0, label_ids)
        decoded_labels = tokenizer.batch_decode(new_label_ids, skip_special_tokens=True)
        golden_questions_array+=decoded_labels
        paragraphs_array+=paragraph

    for i in range(len(generated_beam_array)):
        writer.writerow({'paragraph': paragraphs_array[i], 'golden question': golden_questions_array[i],\
             'generated question - beam search': generated_beam_array[i], 'generated question - top k and p': generated_top_p_array[i]})

        # golden_reference_q.append(golden_question.split(' '))
        # generated_q.append(generated_question.split(' '))

    csv_file.close()
    # bleu, precisions, bp, ratio, question_length, reference_length = compute_bleu(
    #     golden_reference_q, generated_q, max_order=4)
    # print("BLEU: %f" % (bleu))
    # print("Precisions: ", precisions)
    # print("Brevity Penalty: %f" % (bp))
    # print("Ratio: %f" % (ratio))
    # print("Question length: %f" % (question_length))
    # print("Reference length: %f" % (reference_length))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str,
                        default='./config_modified.json', help='QG configuration file')
    parser.add_argument('--data_file', type=str, required=True,
                        help='File path of type tfrecord')
    parser.add_argument('--beam_search', type=bool, default=False, required=True,
                        help='enable beam search startegy')
    parser.add_argument('--top_k', type=bool, default=False, required=True,
                        help='enable top_k decoding startegy')                                                          
    args = parser.parse_args()
    evaluate(args.config_file, args.data_file, args.beam_search, args.top_k)
