import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch import Trainer, seed_everything
from data.SMNIST import SMNIST
from data.BRAILLE import BRAILLE
from network_engine import NetworkEngine
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary
import torch
from extract_statistics import parameter_statistic

torch.set_float32_matmul_precision('high')

if __name__ == '__main__':
    seed_everything(42)

    model_path = "/home/leto/nice_2025/model_insights/results/best_model/braille_exploration_l2mu/best.ckpt"
    model = NetworkEngine.load_from_checkpoint(model_path)
    data_module = BRAILLE(data_dir="../data/braille_splitted", batch_size=model.hparams.params["batch_size"])

    parameter_statistic(model_path)

    trainer = Trainer(accelerator='gpu', devices=[0], num_sanity_val_steps=0, enable_progress_bar=True, logger=None)
    test_data = trainer.test(model=model, datamodule=data_module, verbose=False)[0]
    test_accuracy = test_data['test_accuracy']
    print(test_accuracy)

