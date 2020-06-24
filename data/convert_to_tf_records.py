import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
import json
import re
import os
import argparse


def normalize_text(text):
    text = text.replace('-RRB-', '')
    text = text.replace('-LRB-', '')
    text = re.sub('[\\[|(|)||"|#|/|@|;|:|<|>|{|}|`|+|=|~\||!|]|_|\\]', '<>',
                  text)
    index = text.find('<>')
    # the following postprocessing to handle cases like "here i'm ....."
    while index > -1:
        if index > 0 and text[index - 1] == ' ' or index + 2 < len(
                text) and text[index + 2] == ' ':
            text = text.replace('<>', '', 1)
        else:
            text = text.replace('<>', ' ', 1)
        index = text.find('<>')
    return text.strip()


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy(
        )  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def serialize_example(topic, paragraph, answer, question):
    """
  Creates a tf.Example message ready to be written to a file.
  """
    # Create a dictionary mapping the feature name to the tf.Example-compatible
    # data type.
    feature = {
        'topic': _bytes_feature(topic.encode()),
        'paragraph': _bytes_feature(paragraph.encode()),
        'answer': _bytes_feature(answer.encode()),
        'question': _bytes_feature(question.encode())
    }

    # Create a Features message using tf.train.Example.

    example_proto = tf.train.Example(features=tf.train.Features(
        feature=feature))
    return example_proto.SerializeToString()


def write_tf_records(examples, output_file):
    with tf.io.TFRecordWriter(output_file) as writer:
        for example in examples:
            writer.write(example)


def trim_paragraph(paragraph_text, answer_text):
    paragraph_sentences = nltk.sent_tokenize(paragraph_text)
    answer_sentences = nltk.sent_tokenize(answer_text)

    for i in range(len(paragraph_sentences)):
        sentence = paragraph_sentences[i]
        if len(answer_sentences) > 1 and answer_sentences[0] in sentence:
            #Asumption: if the tokenizer splitted the answer into more than two sentences, it means that the same applied on paragraph text on the same text segment
            answer_found = True
            for j in range(1, len(answer_sentences)):
                if answer_sentences[j] not in paragraph_sentences[i + j]:
                    answer_found = False
                    break
            if answer_found:
                answer_sentence_offset = paragraph_text.find(paragraph_sentences[i + len(answer_sentences) - 1])
                return paragraph_text[:answer_sentence_offset + len(paragraph_sentences[i + len(answer_sentences) - 1])]
                    

        else:    
            if answer_text in sentence:
                answer_sentence_offset = paragraph_text.find(sentence)
                if answer_sentence_offset == -1:
                    print(
                        'Error: something went wrong. can\'t find answer sentence in the paragraph text'
                    )
                    exit(1)

                return paragraph_text[:answer_sentence_offset + len(sentence)]
    print('Error: something went wrong. can\'t find answer in the paragraph')
    exit(1)


def convert_to_tf_records(input_file, trim_up_to_answer, replace_answer_span):
    examples_list = []
    with open(input_file) as json_file:
        for json_example in json_file:
            record = json.loads(json_example)
            paragraph = normalize_text(record['passage'])
            question = normalize_text(record['question'])
            if record['answer']:
                answer = 'true'
            else:
                answer = 'false'
            title = normalize_text(record['title'])
        
            examples_list.append(serialize_example(title, paragraph, answer, question))

    dataset_len = len(examples_list)
    examples_numpy_arr = np.array(examples_list)
    np.random.shuffle(examples_numpy_arr)
    train_examples = examples_numpy_arr[0:int(0.9 * dataset_len)]
    print("train_len ", len(train_examples))
    dev_examples = examples_numpy_arr[int(0.9 * dataset_len)::]
    print("dev_len ", len(dev_examples))                                                           
    splitted_input_file = input_file.split('/')

    train_output_file = '/'.join(
        splitted_input_file[0:len(splitted_input_file) -
                            1]) + '/boolean_qa_train_dataset.tfrecord'
    dev_output_file = '/'.join(
        splitted_input_file[0:len(splitted_input_file) -
                            1]) + '/boolean_qa_dev_dataset.tfrecord'

    write_tf_records(train_examples, train_output_file)
    write_tf_records(dev_examples, dev_output_file)

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path',
                        type=str,
                        help='path to data file',
                        required=True)
    parser.add_argument(
        '--trim_up_to_answer',
        type=str2bool,
        help='boolean flag whether to trim up to answer sentence',
        default=False,
        required=True)
    parser.add_argument(
        '--replace_answer_span',
        type=str2bool,
        default=False,
        help='boolean flag whether to replace answer span by <ANS> tag',
        required=True)
    args = parser.parse_args()
    input_file = args.file_path
    trim_up_to_answer = args.trim_up_to_answer
    replace_answer_span = args.replace_answer_span
    assert os.path.exists(input_file) is not None, "%s file is not found" % (
        input_file)
    convert_to_tf_records(input_file, trim_up_to_answer, replace_answer_span)


if __name__ == '__main__':
    main()