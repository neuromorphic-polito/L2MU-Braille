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
from neurobench.benchmarks import Benchmark
from neurobench.postprocessing import choose_max_count
from neurobench.models import SNNTorchModel

torch.set_float32_matmul_precision('high')

if __name__ == '__main__':
    seed_everything(42)

    experiment_name = "braille_full_exploration_l2mu"

    model_path = f"../model_insights/results/best_model/{experiment_name}/best.ckpt"
    model = NetworkEngine.load_from_checkpoint(model_path)
    data_module = BRAILLE(data_dir="../data/braille_full_splitted", batch_size=model.hparams.params["batch_size"])

    parameter_statistic(model_path)

    trainer = Trainer(accelerator='gpu', devices=[0], num_sanity_val_steps=0, enable_progress_bar=True, logger=False)
    test_data = trainer.test(model=model, datamodule=data_module, verbose=False)[0]
    test_accuracy = test_data['test_accuracy']
    print(test_accuracy)

    model_wrapped = SNNTorchModel(model.model)
    test_set_loader = data_module.test_dataloader()

    preprocessors = []
    postprocessors = [choose_max_count]

    static_metrics = ["footprint", "connection_sparsity"]
    workload_metrics = ["classification_accuracy", "activation_sparsity", "synaptic_operations", "membrane_updates"]

    benchmark = Benchmark(model_wrapped, test_set_loader, preprocessors, postprocessors, [static_metrics, workload_metrics])
    results = benchmark.run()
    print(results)

