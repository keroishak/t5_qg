import tensorflow as tf
import t5
import os
import json

with open('qg_config.json', 'r') as f:
    config = json.load(f)

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

model.predict(
    input_file="data/examples.txt",
    output_file="data/predicted_output.txt",
    temperature=0
)