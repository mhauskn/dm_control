from trainer import Trainer, TrainerConfig
from model import FFNet
from dataset import TrajectoryDataset
from absl import app

def train(argv):
    train_dataset = TrajectoryDataset('small_trajectory_dataset.hdf5', block_size=1)
    model = FFNet()
    tconf = TrainerConfig(max_epochs=10)
    trainer = Trainer(model, train_dataset, None, tconf)
    trainer.train()

if __name__ == "__main__":
    app.run(train)