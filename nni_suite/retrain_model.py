import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch import Trainer, seed_everything
from data.SMNIST import SMNIST
from data.BRAILLE import BRAILLE
from network_engine import NetworkEngine
from lightning.pytorch.callbacks import ModelCheckpoint
import torch

torch.set_float32_matmul_precision('high')

if __name__ == '__main__':
    seed_everything(42)

    params = {'lr': 0.0014500000000000001, 'batch_size': 64, 'order': 7.0, 'theta': 10.0, 'hidden_size': 120.0, 'memory_size': 70.0, 'beta_spk_u': 0.8, 'threshold_spk_u': 0.2, 'beta_spk_h': 0.45, 'threshold_spk_h': 0.65, 'beta_spk_m': 0.2, 'threshold_spk_m': 0.4, 'beta_spk_output': 0.6000000000000001, 'threshold_spk_output': 0.15000000000000002}
    data_module = BRAILLE(data_dir="../data/braille_splitted", batch_size=params['batch_size'], split=9)

    num_inputs = data_module.num_inputs
    num_outputs = data_module.num_outputs

    model = NetworkEngine(num_inputs=num_inputs, num_outputs=num_outputs,
                          params=params, architecture='L2MU')

    trainer = Trainer(accelerator='gpu', devices=[2], max_epochs=300, num_sanity_val_steps=0, enable_progress_bar=True,
                      enable_checkpointing=True,
                      logger=TensorBoardLogger(save_dir="./tensorboard", name="l2mu", version="full"),
                      callbacks=[ModelCheckpoint(monitor='val_accuracy', mode='max', filename='best'),ModelCheckpoint(filename='last')]
                      )
    trainer.fit(model=model, datamodule=data_module)
    test_data = trainer.test(model=model, datamodule=data_module, verbose=False, ckpt_path='best')[0]
    test_accuracy = test_data['test_accuracy']
    print(test_accuracy)

