import tensorflow as tf
from dataset_pipeline import DatasetPipeline
from model import T5QGModel
import argparse
import json


def calculate_loss(labels, logits):
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
    # make sure only labels that are not equal to -100
    # are taken into account as loss
    active_loss = tf.reshape(labels, (-1,)) != -100
    reduced_logits = tf.boolean_mask(tf.reshape(
        logits, (-1, tf.shape(logits)[2])), active_loss)
    labels = tf.boolean_mask(tf.reshape(labels, (-1,)), active_loss)
    return loss_fn(labels, reduced_logits)


def get_callbacks(checkpoint_path):
    if checkpoint_path[-1] == '/':
        checkpoint_path = checkpoint_path[:-1]

    model_checkpoint = checkpoint_path + '/t5_qg.ckpt'
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_checkpoint, save_weights_only=True, verbose=1, period=2),
        tf.keras.callbacks.TensorBoard(log_dir='./runs'),
        tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss')
    ]


def train(qg_config_path):
    with open(qg_config_path, 'r') as f:
        config = json.load(f)

    batch_size = config['batch_size']
    train_ds = config['train_ds_file']
    val_ds = config['dev_ds_file']
    sentencepiece_mode_path = config['sentencepiece_model_path']
    target_max_len = config['target_max_len']

    train_input = DatasetPipeline(
        train_ds, batch_size, sentencepiece_mode_path, target_max_len, True, slim_dataset=True)
    val_input = DatasetPipeline(
        val_ds, batch_size, sentencepiece_mode_path, target_max_len, False, slim_dataset=True)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        config['initial_lr'], decay_steps=config['decay_steps'], decay_rate=0.96, staircase=True)

    optimizer = tf.keras.optimizers.Adam(lr_schedule, beta_1=0.9, beta_2=0.98,
                                         epsilon=1e-9)
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    model_dir = config['t5_model_dir']
    qg = T5QGModel(model_dir)

    if config['restore']:
        latest = tf.train.latest_checkpoint(config['checkpoint_path'])
        qg.model.load_weights(latest)

    qg.model.compile(optimizer=optimizer,
                     loss=calculate_loss,
                     metrics=[train_accuracy])

    qg.model.fit(train_input.dataset,
                 epochs=config['epochs'],
                 callbacks=get_callbacks(config['checkpoint_path']),
                 validation_data=val_input.dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str,
                        default='./config.json', help='QG configuration file')
    args = parser.parse_args()
    train(args.config_file)
