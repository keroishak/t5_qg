import tensorflow as tf
import tensorflow_probability as tfp
from dataset_pipeline import DatasetPipeline, ProcessMode
from model import T5QGModel
from gradient_accumulator import GradientAccumulator
import argparse
import json
import numpy as np

tf.config.run_functions_eagerly(True)
recompute = tf.constant(87599) # this number should represent the dataset size
last_computed = tf.Variable(0)
percentile_val = tf.Variable(1e9)
k = tf.Variable(3500)
top_k_vals = tf.Variable([])

@tf.function
def get_mask2(loss, top_k_vals, k, last_computed, percentile_val):
    loss_len = tf.size(loss)
    last_computed_fun = last_computed
    last_computed_fun.assign_add(loss_len)


    top_k_candidates = tf.concat([top_k_vals, loss], 0)
    if tf.size(top_k_candidates) > k:
        top_k_candidates = tf.math.top_k(top_k_candidates, k=k, sorted=False).values

    if last_computed_fun >= recompute:
        percentile_val_fun = tf.math.reduce_min(top_k_candidates)
        last_computed_fun.assign(0)
        top_k_candidates = tf.Variable([])
        # tf.print(percentile_val_fun)
        # tf.print(tf.size(top_k_candidates))
        mask = loss < percentile_val_fun
        return mask, top_k_candidates, last_computed_fun, percentile_val_fun
    else:    
        mask = loss < percentile_val
        return mask, top_k_candidates, last_computed_fun, percentile_val
    


def calculate_loss(labels, logits):
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
    # make sure only labels that are not equal to -100
    # are taken into account as loss
    nonnegative_mask = labels != -100
    labels = labels * tf.cast(nonnegative_mask, dtype=tf.int64)
    __loss = loss_fn(labels, logits)
    batch_loss = __loss * tf.cast(nonnegative_mask, dtype=tf.float32)
    batch_loss = tf.math.reduce_mean(batch_loss, axis=1)
    global top_k_vals, k, last_computed, percentile_val
    mask, top_k_candidates, last_computed_fun, percentile_val_fun = get_mask2(batch_loss, top_k_vals, k, last_computed, percentile_val)
    last_computed.assign(last_computed_fun)
    percentile_val.assign(percentile_val_fun)
    top_k_vals = top_k_candidates[:]

    mask_in_float = tf.cast(mask, dtype=__loss.dtype)
    if tf.reduce_sum(mask_in_float) == 0:
        missed_batches.assign_add(1)
        return 0
    final_loss = batch_loss * mask_in_float
    return tf.reduce_sum(final_loss) / tf.reduce_sum(mask_in_float)

def get_callbacks(checkpoint_path, model_checkpoint_name):
    if checkpoint_path[-1] == '/':
        checkpoint_path = checkpoint_path[:-1]

    model_checkpoint = checkpoint_path + '/'+ model_checkpoint_name
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_checkpoint, save_weights_only=True, verbose=1, period=2),
        tf.keras.callbacks.TensorBoard(log_dir='./runs'),
        # tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss')
    ]


def train(qg_config_path):
    with open(qg_config_path, 'r') as f:
        config = json.load(f)

    batch_size = config['batch_size']
    train_ds = config['train_ds_file']
    val_ds = config['dev_ds_file']
    sentencepiece_mode_path = config['sentencepiece_model_path']
    
    model_checkpoint_name = 't5_qg_without_dropping.ckpt'

    train_input = DatasetPipeline(
        train_ds, batch_size, sentencepiece_mode_path, ProcessMode.Training)   
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        config['initial_lr'], decay_steps=config['decay_steps'], decay_rate=0.96, staircase=True)

    optimizer = tf.keras.optimizers.Adam(lr_schedule, beta_1=0.9, beta_2=0.98,
                                         epsilon=1e-9)
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    model_dir = config['t5_model_dir']
    qg = T5QGModel(model_dir, train_input.get_tokenizer_size())

    if config['restore']:
        latest = tf.train.latest_checkpoint(config['checkpoint_path'])
        qg.model.load_weights(latest)

    qg.model.compile(optimizer= GradientAccumulator(optimizer),
                     loss=calculate_loss,
                     metrics=[train_accuracy])

    qg.model.fit(train_input.dataset,
                 epochs=config['epochs'],
                 callbacks=get_callbacks(config['checkpoint_path'], model_checkpoint_name),
                #  validation_data=val_input.dataset
                 )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str,
                        default='./config.json', help='QG configuration file')
    args = parser.parse_args()
    train(args.config_file)
