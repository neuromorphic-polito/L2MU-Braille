import os
import sys
from platform import architecture

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from lightning.pytorch import Trainer, seed_everything
from data.SMNIST import SMNIST
from data.BRAILLE import BRAILLE
from network_engine import NetworkEngine
import torch

torch.set_float32_matmul_precision('high')

if __name__ == '__main__':
    seed_everything(42)

    params = {"lr":0.005,"batch_size":256.0,"order":18.0,"theta":13.4,"hidden_size":180.0,"memory_size":2,"beta_spk_u":0.4,"threshold_spk_u":0.015,"beta_spk_h":0.2,"threshold_spk_h":0.65,"beta_spk_m":0.55,"threshold_spk_m":0.9,"beta_spk_output":0.7,"threshold_spk_output":0.75}
    #dataset_path = "../data/shd"
    data_module = BRAILLE(batch_size=params['batch_size'])

    num_inputs = data_module.num_inputs
    num_outputs = data_module.num_outputs

    model = NetworkEngine(num_inputs=num_inputs, num_outputs=num_outputs,
                          params=params, architecture='L2MU')

    trainer = Trainer(accelerator='gpu', devices=[0], max_epochs=10, num_sanity_val_steps=0, enable_progress_bar=True,
                      enable_checkpointing=True,
                      logger=False,
                      )
    trainer.fit(model=model, datamodule=data_module)
    test_data = trainer.test(model=model, datamodule=data_module, verbose=False)[0]
    # model.to_onnx(export_params=True)
    test_accuracy = test_data['test_accuracy']
    print(test_accuracy)

