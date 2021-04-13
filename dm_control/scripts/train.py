import torch
import os
import numpy as np
from trainer import Trainer, TrainerConfig
from model import FFNet, GPT, GPTConfig
from dataset import TrajectoryDataset
from absl import flags, logging, app

OUTPUT_DIR = os.environ.get('PT_OUTPUT_DIR', '.')
DATA_DIR = os.environ.get('PT_DATA_DIR', 'data')

FLAGS = flags.FLAGS
flags.DEFINE_string("checkpoint_path", 'saved_model.pt', 'Path to save model checkpoints')
flags.DEFINE_string("config_path", 'saved_model_config.json', 'Path to save model config')
flags.DEFINE_string("dataset", 'single_episode.hdf5', 'Name of dataset (assumed inside of data_dir).')
flags.DEFINE_integer("max_epochs", 10, "Maximum training epochs.")
flags.DEFINE_integer("batch_size", 64, "Batch size used during training.")
flags.DEFINE_float("learning_rate", .0001, "Learning rate")
flags.DEFINE_float("grad_norm_clip", 5.0, "Clip Gradient Norm")
flags.DEFINE_integer("block_size", 4, "Size of history/context used.")
flags.DEFINE_integer("gpt_layers", 8, "Number of layers in GPT Model")
flags.DEFINE_integer("gpt_heads", 8, "Number of heads in GPT Model")
flags.DEFINE_integer("gpt_embd", 512, "Size of GPT Embed")
flags.DEFINE_list("observables", "joints_pos, joints_vel", "List of observation features to use.")

def train():
    train_dataset = TrajectoryDataset(
        h5py_file=os.path.join(DATA_DIR, FLAGS.dataset),
        block_size=FLAGS.block_size,
        observables=FLAGS.observables,
    )
    mconf = GPTConfig(
        obs_size=train_dataset.observation_size, 
        action_size=train_dataset.action_size, 
        block_size=train_dataset.block_size,
        n_layer=FLAGS.gpt_layers,
        n_head=FLAGS.gpt_heads, 
        n_embd=FLAGS.gpt_embd,
    )
    # Save the config to a json file
    mconf.to_json(os.path.join(OUTPUT_DIR, FLAGS.config_path))
    model = GPT(mconf)
    tconf = TrainerConfig(
        batch_size=FLAGS.batch_size,
        max_epochs=FLAGS.max_epochs, 
        ckpt_path=os.path.join(OUTPUT_DIR, FLAGS.checkpoint_path),
        learning_rate=FLAGS.learning_rate,
        betas=(0.9, 0.999),
        grad_norm_clip=FLAGS.grad_norm_clip,
        lr_decay=False,
    )
    trainer = Trainer(model, train_dataset, None, tconf)
    trainer.train()
    return model

def log_flags(flags):
    """ Logs the value of each of the flags. """
    for k in dir(flags):
        if k != '?':
            flag = 'FLAGS.{}'.format(k)
            logging.info('{}: {}'.format(flag, eval(flag)))

def main(argv):
    log_flags(FLAGS)
    model = train()

if __name__ == "__main__":
    app.run(main)