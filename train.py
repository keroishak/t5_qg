import tensorflow as tf
from dataset_pipeline import DatasetPipeline
from model import T5QGModel
import argparse
import json


def get_loss_function():
    return tf.keras.losses.SparseCategoricalCrossentropy(reduction='none')


def calculate_loss(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_object = get_loss_function()
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


def get_callbacks(config):
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=config['checkpoint_path'], save_best_only=True, monitor='val_sparse_categorical_accuracy', save_weights_only=True, verbose=1, period=2),
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

    train_input = DatasetPipeline(
        train_ds, batch_size, sentencepiece_mode_path)
    val_input = DatasetPipeline(
        val_ds, batch_size, sentencepiece_mode_path)

    # learning_rate = CustomSchedule(
    #     self.config['hidden_size'], warmup_steps=self.config['warmup_steps'])
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        config['initial_lr'], decay_steps=config['decay_steps'], decay_rate=0.96, staircase=True)
    # # self.optimizer = tf.keras.optimizers.SGD(learning_rate)
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
                 callbacks=get_callbacks(config),
                 validation_data=val_input.dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str,
                        default='./config.json', help='QG configuration file')
    args = parser.parse_args()
    train(args.config_file)
