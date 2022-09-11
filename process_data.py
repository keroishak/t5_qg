import tensorflow as tf
import numpy as np
import json
import re
import os
import argparse
import nltk
from tqdm import tqdm
from transformers import T5TokenizerFast

nltk.download("punkt")


def convert_idx(text, tokens):
    current = 0
    spans = []
    for token in tokens:
        current = text.find(token, current)
        if current < 0:
            print("Token {} cannot be found".format(token))
            raise Exception()
        spans.append((current, current + len(token)))
        current += len(token)

    return spans


def tokenize(doc, parser):
    words = []
    sentences = parser(doc).sentences
    for sent in sentences:
        toks = sent.tokens
        for token in toks:
            words.append(token.text.lower())
    return words


def word_tokenize(tokens):
    return [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)]


def get_dataset_examples(file_name):
    examples = list()
    total = 0
    with open(file_name, "r") as f:
        source = json.load(f)
        articles = source["data"]
    for article in tqdm(articles):
        for para in article["paragraphs"]:
            context = para["context"].replace(
                "''", '" ').replace("``", '" ').lower()

            context_tokens = word_tokenize(context)
            spans = convert_idx(context, context_tokens)

            for qa in para["qas"]:
                total += 1
                ques = qa["question"].replace(
                    "''", '" ').replace("``", '" ').lower()
                ques_tokens = word_tokenize(ques)

                y1s, y2s = [], []
                answer_texts = []

                for answer in qa["answers"]:
                    answer_text = answer["text"].lower()
                    answer_start = answer["answer_start"]
                    answer_end = answer_start + len(answer_text)
                    answer_texts.append(answer_text)
                    answer_span = []

                    for idx, span in enumerate(spans):
                        if not (answer_end <= span[0] or answer_start >= span[1]):
                            answer_span.append(idx)
                    y1, y2 = answer_span[0], answer_span[-1]
                    y1s.append(y1)
                    y2s.append(y2)
                example = {"context_tokens": context_tokens, "ques_tokens": ques_tokens,
                           "y1s": y1s, "y2s": y2s, "answers": answer_texts}
                examples.append(example)

    return examples

def get_dataset_examples_v2(file_name, tokenizer, start_answer_special_token, end_answer_special_token):
    examples = list()
    total = 0
    with open(file_name, "r") as f:
        source = json.load(f)
        articles = source["data"]
    for article in tqdm(articles):
        for para in article["paragraphs"]:
            context = para["context"].replace(
                "''", '" ').replace("``", '" ').lower()

            context_tokens = word_tokenize(context)
            spans = convert_idx(context, context_tokens)

            for qa in para["qas"]:
                total += 1
                ques = qa["question"].replace(
                    "''", '" ').replace("``", '" ').lower()
                ques_tokens = word_tokenize(ques)

                y1s, y2s = [], []
                answer_texts = []

                for answer in qa["answers"]:
                    answer_text = answer["text"].lower()
                    answer_start = answer["answer_start"]
                    answer_end = answer_start + len(answer_text)
                    answer_texts.append(answer_text)
                    answer_span = []

                    for idx, span in enumerate(spans):
                        if not (answer_end <= span[0] or answer_start >= span[1]):
                            answer_span.append(idx)
                    y1, y2 = answer_span[0], answer_span[-1]
                    y1s.append(y1)
                    y2s.append(y2)
                answer_start_idx = y1s[0]
                answer_end_idx = y2s[0]
                paragraph_with_answer_markers = context_tokens[:answer_start_idx] + [start_answer_special_token] + context_tokens[answer_start_idx:answer_end_idx+1] + [end_answer_special_token] + context_tokens[answer_end_idx+1:]
                input_text = ' '.join(paragraph_with_answer_markers)
                labels = ' '.join(ques_tokens)
                task_prefix = "qg_generation:"
                encoded_input = tokenizer.encode_plus(task_prefix + input_text, max_length=512, padding='max_length', truncation=True, return_tensors='np')
                encoded_labels = tokenizer.encode_plus(labels, max_length=60, padding='max_length', truncation=True, return_tensors='np')
                
                encoded_labels_ids = encoded_labels["input_ids"][0]
                encoded_labels_ids[encoded_labels_ids == tokenizer.pad_token_id] = -100

                encoded_labels_mask = encoded_labels['attention_mask'][0]
                example = {"input_ids":encoded_input["input_ids"][0], "label_ids": encoded_labels_ids , 'decoder_attention_mask': encoded_labels_mask}
                examples.append(example)

    return examples

def to_tf_records(examples):
    tf_records_list = []
    for example in examples:
        paragraph = ' '.join(example['context_tokens'])
        question = ' '.join(example['ques_tokens'])
        answers = '|'.join(example['answers'])
        answer_start_idx = example['y1s'][0]
        answer_end_idx = example['y2s'][0]

        tf_example = serialize_example(paragraph, answers, question, answer_start_idx, answer_end_idx)
        tf_records_list.append(tf_example)
    return tf_records_list

def to_tf_records_v2(examples):
    tf_records_list = []
    for example in examples:
        input_ids = example['input_ids']
        label_ids = example['label_ids']
        decoder_attention_mask = example['decoder_attention_mask']
        tf_example = serialize_example_v2(input_ids, label_ids, decoder_attention_mask)
        tf_records_list.append(tf_example)
    return tf_records_list

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

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _int64_array_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def serialize_example(paragraph, answer, question, answer_start_idx, answer_end_idx):
    """
  Creates a tf.Example message ready to be written to a file.
  """
    # Create a dictionary mapping the feature name to the tf.Example-compatible
    # data type.
    feature = {
        'paragraph': _bytes_feature(paragraph.encode()),
        'answer': _bytes_feature(answer.encode()),
        'question': _bytes_feature(question.encode()),
        'answer_start_idx': _int64_feature(answer_start_idx),
        'answer_end_idx': _int64_feature(answer_end_idx)
    }

    # Create a Features message using tf.train.Example.

    example_proto = tf.train.Example(features=tf.train.Features(
        feature=feature))
    return example_proto.SerializeToString()

def serialize_example_v2(input_ids, label_ids, decoder_attention_mask):
    """
  Creates a tf.Example message ready to be written to a file.
  """
    # Create a dictionary mapping the feature name to the tf.Example-compatible
    # data type.
    feature = {
        'input_ids': _int64_array_feature(input_ids),
        'label_ids': _int64_array_feature(label_ids),
        'decoder_attention_mask': _int64_array_feature(decoder_attention_mask)
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
            # Asumption: if the tokenizer splitted the answer into more than two sentences, it means that the same applied on paragraph text on the same text segment
            answer_found = True
            for j in range(1, len(answer_sentences)):
                if answer_sentences[j] not in paragraph_sentences[i + j]:
                    answer_found = False
                    break
            if answer_found:
                answer_sentence_offset = paragraph_text.find(
                    paragraph_sentences[i + len(answer_sentences) - 1])
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

def process_file(train_file_name, dev_file_name, output_path, trim_up_to_answer):
    if output_path[-1] == '/':
        output_path = output_path[:-1]
    train_examples = get_dataset_examples(train_file_name)
    train_tf_record_examples = to_tf_records(train_examples)
    train_output_file = output_path + '/train_dataset.tfrecord'
    print("Training examples length: ", len(train_tf_record_examples))
    write_tf_records(train_tf_record_examples, train_output_file)

    dev_examples = get_dataset_examples(dev_file_name)
    tf_record_examples = to_tf_records(dev_examples)
    num_dev = len(tf_record_examples) // 2
    dev_examples = tf_record_examples[:num_dev]
    test_examples = tf_record_examples[num_dev:]

    dev_output_file = output_path + '/dev_dataset.tfrecord'
    test_output_file = output_path + '/test_dataset.tfrecord'

    print("Development examples length: ", len(dev_examples))
    write_tf_records(dev_examples, dev_output_file)

    print("Testing examples length: ", len(test_examples))
    write_tf_records(test_examples, test_output_file)


def convert_to_tf_records(is_train, input_file, trim_up_to_answer, replace_answer_span):
    examples_list = []
    with open(input_file) as json_file:
        data = json.load(json_file)
        topic_list = data['data']
        unlabeled_answer_segment = 0
        for topic in topic_list:
            topic_title = topic['title']
            topic_title_text = normalize_text(topic_title)
            for paragraph in topic['paragraphs']:
                paragraph_text = normalize_text(paragraph['context'])
                for question in paragraph['qas']:
                    question_text = normalize_text(question['question'])
                    answer = question['answers'][0]
                    answer_text = normalize_text(answer['text'])
                    trimmed_paragraph_text = paragraph_text
                    if trim_up_to_answer:
                        trimmed_paragraph_text = trim_paragraph(
                            paragraph_text, answer_text)
                    examples_list.append(
                        serialize_example(topic_title_text,
                                          trimmed_paragraph_text, answer_text,
                                          question_text))
    dataset_len = len(examples_list)
    examples_numpy_arr = np.array(examples_list)
    splitted_input_file = input_file.split('/')
    if is_train:
        np.random.shuffle(examples_numpy_arr)
        train_examples = examples_numpy_arr[0:int(0.9 * dataset_len)]
        print("train_len ", len(train_examples))
        dev_examples = examples_numpy_arr[int(0.9 * dataset_len)::]
        print("dev_len ", len(dev_examples))
        # test_examples = examples_numpy_arr[int(0.9 * dataset_len)::]
        # print("test_len ", len(test_examples))
        train_output_file = '/'.join(
            splitted_input_file[0:len(splitted_input_file) -
                                1]) + '/train_datasett.tfrecord'
        dev_output_file = '/'.join(
            splitted_input_file[0:len(splitted_input_file) -
                                1]) + '/dev_datasett.tfrecord'
        # test_output_file = '/'.join(
        #     splitted_input_file[0:len(splitted_input_file) -
        #                         1]) + '/test_dataset.tfrecord'

        write_tf_records(train_examples, train_output_file)
        write_tf_records(dev_examples, dev_output_file)
        # write_tf_records(test_examples, test_output_file)
    else:
        test_output_file = '/'.join(
            splitted_input_file[0:len(splitted_input_file) -
                                1]) + '/test_datasett.tfrecord'
        write_tf_records(examples_numpy_arr, test_output_file)

    # special_tokens = ['<PAD>', '<SEP>', '<SOS>', '<EOS>', '<ANS>']
    # vocab_text = []
    # for row in training_examples:
    #     example = tf.train.Example()
    #     example.ParseFromString(row)
    #     vocab_text.append(
    #         example.features.feature['paragraph'].bytes_list.value[0].decode(
    #             'utf-8'))
    # tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    #     corpus_generator=(txt for txt in vocab_text),
    #     reserved_tokens=special_tokens,
    #     target_vocab_size=2**13)
    # tokenizer.save_to_file('./wordpieces')
    

def convert_file_into_tensorized_features(train_file_name, dev_file_name, output_path):
    start_answer_special_token = "<ANS>"
    end_answer_special_token = "</ANS>"
    tokenizer = T5TokenizerFast.from_pretrained("t5-small")
    special_tokens_dict = {'additional_special_tokens': [start_answer_special_token, end_answer_special_token]}
    
    tokenizer.add_special_tokens(special_tokens_dict)

    if output_path[-1] == '/':
        output_path = output_path[:-1]
    train_examples = get_dataset_examples_v2(train_file_name, tokenizer, start_answer_special_token, end_answer_special_token)
    train_tf_record_examples = to_tf_records_v2(train_examples)
    train_output_file = output_path + '/train_dataset_v2.tfrecord'
    print("Training examples length: ", len(train_tf_record_examples))
    write_tf_records(train_tf_record_examples, train_output_file)

    dev_examples = get_dataset_examples_v2(dev_file_name, tokenizer, start_answer_special_token, end_answer_special_token)
    tf_record_examples = to_tf_records_v2(dev_examples)
    num_dev = len(tf_record_examples) // 2
    dev_examples = tf_record_examples[:num_dev]
    test_examples = tf_record_examples[num_dev:]

    dev_output_file = output_path + '/dev_dataset_v2.tfrecord'
    test_output_file = output_path + '/test_dataset_v2.tfrecord'

    print("Development examples length: ", len(dev_examples))
    write_tf_records(dev_examples, dev_output_file)

    print("Testing examples length: ", len(test_examples))
    write_tf_records(test_examples, test_output_file)

    print("Saving tokenizer ...")
    tokenizer.save_pretrained(output_path + '/tokenizer')

        

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
    parser.add_argument('--train_file_path', type=str,
                        help='path to SQuAD train JSON file',
                        required=True)
    parser.add_argument('--dev_file_path', type=str,
                        help='path to SQuAD dev JSON file',
                        required=True)
    parser.add_argument('--output_path', type=str,
                        help='output path',
                        required=True)

    parser.add_argument('--trim_up_to_answer', type=str2bool,
                        help='boolean flag whether to trim up to answer sentence',
                        default=False,
                        required=True)
    # parser.add_argument('--replace_answer_span', type=str2bool,
    #                     default=False,
    #                     help='boolean flag whether to replace answer span by <ANS> tag',
    #                     required=True)
    args = parser.parse_args()
    # is_train = args.train
    train_file_path = args.train_file_path
    dev_file_path = args.dev_file_path
    output_path = args.output_path
    trim_up_to_answer = args.trim_up_to_answer
    # replace_answer_span = args.replace_answer_span
    # TODO: assert on file location
    # assert os.path.exists(input_file) is not None, "%s file is not found" % (
    #     input_file)
    convert_file_into_tensorized_features(train_file_path, dev_file_path, output_path)


if __name__ == '__main__':
    main()
