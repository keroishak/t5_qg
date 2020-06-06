import tensorflow as tf
import t5
import os
import json

with open('qg_config.json', 'r') as f:
    config = json.load(f)

qg_tfrecords_path = {
    "train": config['train_ds_file'],
    "validation": config['dev_ds_file']
}
def encode(txt_record):
    example = tf.train.Example()
    example.ParseFromString(txt_record.numpy())
    paragraph = example.features.feature['paragraph'].bytes_list.value[
        0].decode('utf-8')
    answer = example.features.feature['answer'].bytes_list.value[0].decode(
        'utf-8')
    question = example.features.feature['question'].bytes_list.value[
        0].decode('utf-8')

    return  paragraph + " [SEP] " + answer,  question

def __tf_encode(txt_record):
        return tf.py_function(encode, [txt_record],
                              [tf.string, tf.string])
# Create a description of the features.
feature_description = {
    'paragraph': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'answer' : tf.io.FixedLenFeature([], tf.string, default_value=''),
    'question' : tf.io.FixedLenFeature([], tf.string, default_value='')
}

def _parse_function(example_proto):
  # Parse the input `tf.Example` proto using the dictionary above.
  return tf.io.parse_single_example(example_proto, feature_description)

def qg_dataset_fn(split, shuffle_files=False):
    del shuffle_files
    # Load lines from the text file as examples.
    ds = tf.data.TFRecordDataset([qg_tfrecords_path[split]])
    # Split each "<question>\t<answer>" example into (question, answer) tuple.
    # ds = ds.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE).map(lambda x:{"inputs":tf.strings.join([x['paragraph'], x['answer']], separator='</s>'), "targets":x['question']})
    return ds
     

t5.data.TaskRegistry.add(
    "qg_generation",
    # Supply a function which returns a tf.data.Dataset.
    dataset_fn=qg_dataset_fn,
    splits=["train", "validation"],
    # Supply a function which preprocesses text from the tf.data.Dataset.
    text_preprocessor=None,
    output_features = t5.data.Feature(vocabulary=t5.data.SentencePieceVocabulary(config['sentencepiece_model_path'], 100)),
    # Lowercase targets before computing metrics.
    postprocess_fn=t5.data.postprocessors.lower_text, 
    # We'll use accuracy as our evaluation metric.
    metric_fns=[t5.evaluation.metrics.accuracy]
    # Not required, but helps for mixing and auto-caching.
    # num_input_examples=num_nq_examples
)

# The models from our paper are based on the Mesh Tensorflow Transformer.
model = t5.models.MtfModel(
    model_dir=config['t5_model_dir'],
    tpu="",
    mesh_devices = ["gpu:0"],
    mesh_shape = None, #"model:1,batch:15",
    batch_size=config['batch_size'],
    sequence_length={"inputs": 350, "targets": 50},
    learning_rate_schedule=config['initial_lr'],
    save_checkpoints_steps=5000,
    keep_checkpoint_max= None,
    iterations_per_loop=100,
)

model.finetune(
    mixture_or_task_name="qg_generation",
    pretrained_model_dir=config['t5_model_dir'],
    finetune_steps=config['epochs'] * 6666
)