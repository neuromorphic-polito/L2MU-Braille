import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(PARENT_DIR)
GRANDPARENT_DIR = os.path.dirname(PARENT_DIR)
sys.path.append(GRANDPARENT_DIR)

import shutil
import psutil
import torch
import time
from lightning.pytorch import Trainer, seed_everything
from data.SMNIST import SMNIST
from data.BRAILLE import BRAILLE
from network_engine import NetworkEngine
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import nni
from nni.tools.nnictl import updater
import numpy as np
import datetime
import logging
import argparse
from nni_suite.callbacks import ReportMetrics, ValidationDeltaStopping
from nni_suite.utils import SearchSpaceUpdater
from pathlib import Path
import csv

torch.set_float32_matmul_precision('high')

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def configure_trainer(num_epochs, tensorboard_log_dir, gpu_index, nni_report=True):

    callback_training_history = ReportMetrics(nni_report=nni_report)

    callbacks = [callback_training_history]
    if nni_report:
        callbacks.append(ValidationDeltaStopping())

    callbacks.append(ModelCheckpoint(monitor='val_accuracy', mode='max', filename='best'))
    callbacks.append(ModelCheckpoint(filename='last'))

    trainer = Trainer(
        accelerator="gpu",
        max_epochs=num_epochs,
        num_sanity_val_steps=0,
        devices=[0],
        enable_progress_bar=False,
        enable_checkpointing=True,
        logger=TensorBoardLogger(save_dir=tensorboard_log_dir, name="", version=""),
        # ModelCheckPoint(monitor='val_accuracy', mode='max')
        callbacks=callbacks
    )

    return trainer, callback_training_history


def get_checkpoint_file_path(tensorboard_log_dir):
    checkpoint_path = tensorboard_log_dir / "checkpoints"

    if checkpoint_path.is_dir():
        # Find the first file with 'last' in its name
        for file_name_checkpoint in checkpoint_path.iterdir():
            if file_name_checkpoint.is_file() and 'last' in file_name_checkpoint.name:
                logger.debug(f"Checkpoint file found: {file_name_checkpoint}")
                logger.debug(
                    f"Restoring model from checkpoint path: {file_name_checkpoint}"
                )
                return file_name_checkpoint

    return None


def load_model(num_inputs, num_outputs, architecture, params):
    # model info

    # Init model
    return NetworkEngine(
        num_inputs=num_inputs,
        num_outputs=num_outputs,
        architecture=architecture,
        params=params,
    )


def load_data(dataset_path, batch_size):
    # Load dataset
    if "smnist" in dataset_path:
        return SMNIST(dataset_path, batch_size=batch_size)
    elif "braille" in dataset_path:
        return BRAILLE(dataset_path, batch_size=batch_size)
    else:
        raise ValueError("Dataset not found")


def process_train_val_results(training_hist, validation_hist, trainer):
    # best training and validation at best training
    acc_best_train = np.max(training_hist[:, 1])
    epoch_best_train = np.argmax(training_hist[:, 1])
    acc_val_at_best_train = validation_hist[epoch_best_train][1]

    # best validation and training at best validation
    acc_best_val = np.max(validation_hist[:, 1])
    epoch_best_val = np.argmax(validation_hist[:, 1])
    acc_train_at_best_val = training_hist[epoch_best_val][1]

    logger.debug("Trial results: ")
    logger.debug(
        f"\tBest training accuracy: {np.round(acc_best_train * 100, 4)}% "
        f"({np.round(acc_val_at_best_train * 100, 4)}% corresponding validation accuracy) "
        f"at epoch {epoch_best_train + 1}/{trainer.max_epochs}"
    )
    logger.debug(
        f"\tBest validation accuracy: {np.round(acc_best_val * 100, 4)}% "
        f"({np.round(acc_train_at_best_val * 100, 4)}% corresponding training accuracy) "
        f"at epoch {epoch_best_val + 1}/{trainer.max_epochs}"
    )

    return acc_best_train, acc_best_val


def process_test_result(trainer, data_module, model):
    # ckpt='best' if checkpointing callback is passed and based on the best validation
    test_data = trainer.test(
        model=model, datamodule=data_module, verbose=False, ckpt_path='best'
    )[0]
    test_accuracy = test_data["test_accuracy"]
    logger.debug("Test accuracy {}%".format(np.round(test_accuracy * 100, 4)))

    return test_accuracy


def run_experiment(args, params):
    data_module = load_data(
        dataset_path=args.dataset_path, batch_size=int(params["batch_size"])
    )

    model = load_model(
        num_inputs=data_module.num_inputs,
        num_outputs=data_module.num_outputs,
        params=params,
        architecture=args.architecture,
    )

    tensorboard_log_dir = Path(os.path.join(os.environ["NNI_OUTPUT_DIR"], "tensorboard"))
    trainer, callback_training_history = configure_trainer(
        num_epochs=args.num_epochs, gpu_index=args.gpu_index, tensorboard_log_dir=tensorboard_log_dir
    )

    ckpt_file_path = get_checkpoint_file_path(tensorboard_log_dir)

    trainer.fit(model=model, datamodule=data_module, ckpt_path=ckpt_file_path)

    training_results = callback_training_history.training_results
    validation_results = callback_training_history.validation_results

    training_hist = np.array(training_results)
    validation_hist = np.array(validation_results)

    acc_best_train, acc_best_val = process_train_val_results(
        training_hist, validation_hist, trainer
    )

    test_accuracy = process_test_result(trainer, data_module, model)

    nni.report_final_result(
        {
            "default": np.round(
                acc_best_val * 100, 4
            ),  # the default value is the maximum validation accuracy achieved
            "best training": np.round(acc_best_train * 100, 4),
            "test accuracy": np.round(test_accuracy * 100, 4),
        }
    )

    return training_results, validation_results, test_accuracy


def parse_arguments():
    parser = argparse.ArgumentParser()

    # Name for the experiment
    parser.add_argument(
        "-exp_name", type=str, required=True, help="Name for the starting experiment."
    )

    # Which GPU to use
    parser.add_argument(
        "-gpu_index",
        type=int,
        required=True,
        help="GPU index to be used for the experiment.",
    )

    parser.add_argument(
        "-cpu_limit_cores", type=int, required=True, help="Cores limit to use."
    )

    parser.add_argument(
        "-save_weights",
        type=bool,
        required=True,
        help="Weights can be saved to be loaded after training and used for test.",
    )

    # Set seed usage
    parser.add_argument(
        "-seed", type=int, required=True, help="Set if a seed is to be used or not."
    )

    parser.add_argument(
        "-dataset_path", type=str, required=True, help="Path to the dataset"
    )

    parser.add_argument(
        "-architecture",
        type=str,
        required=True,
        help="The network architecture that needs to be trained",
    )

    parser.add_argument("-working_directory",
                        type=str,
                        required=True,
                        help="Path of the working directory.")

    parser.add_argument(
        "-num_epochs",
        type=int,
        required=True,
        help="Number of epochs the model should be trained.",
    )

    return parser.parse_args()


def save_test_results(csv_file_path, test_accuracy):
    with open(csv_file_path, mode="a", newline="") as file:
        fieldnames = ['Test Accuracy', 'Trial ID']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if file.tell() == 0:  # Write header if the file is empty
            writer.writeheader()
        writer.writerow({'Test Accuracy': f"{test_accuracy * 100}%", 'Trial ID': nni.get_trial_id()})


def get_max_accuracy_from_cvs(csv_file_path):

    max_accuracy = -1
    if csv_file_path.exists():
        with open(csv_file_path, mode="r") as file:
            reader = csv.DictReader(file)
            max_accuracy = max(
                [float(row["Test Accuracy"].strip("%")) for row in reader],
                default=0,
            )

    return max_accuracy


def save_best_model(path_best_model, path_candidate_best_model):
    # Ensure the destination directory exists
    if path_best_model.exists():
        shutil.rmtree(str(path_best_model))

    path_best_model.mkdir(parents=True, exist_ok=True)

    for item in path_candidate_best_model.iterdir():
        destination = path_best_model / item.name
        if item.is_dir():
            shutil.copytree(str(item), str(destination))
        else:
            shutil.copy2(str(item), str(destination))


def limit_cpu_cores(cpu_limit_cores):
    if cpu_limit_cores == -1:
        return

    cores_to_use = get_least_active_cores(num_cores=cpu_limit_cores)
    logger.debug(f'Selected CPU cores: {cores_to_use}')

    num_cores = len(cores_to_use)

    pid = os.getpid()  # the current process

    available_cores = list(range(psutil.cpu_count()))
    # selected_cores = available_cores[:num_cores]
    selected_cores = []
    for ii in cores_to_use:
        if ii in available_cores:
            selected_cores.append(ii)

    os.sched_setaffinity(pid, selected_cores)

    # Limit the number of threads used by different libraries
    os.environ["OMP_NUM_THREADS"] = str(num_cores)
    os.environ["MKL_NUM_THREADS"] = str(num_cores)
    torch.set_num_threads(num_cores)


def get_least_active_cores(num_cores, num_readings=10):
    # Get CPU usage for each core for multiple readings
    cpu_usage_readings = []
    for ii in range(num_readings):
        cpu_usage_readings.append(psutil.cpu_percent(percpu=True))
        time.sleep(0.05)

    # Calculate the average CPU usage for each core
    avg_cpu_usage = [sum(usage) / num_readings for usage in zip(*cpu_usage_readings)]

    # Create a list of tuples (core_index, avg_cpu_usage)
    core_usage_tuples = list(enumerate(avg_cpu_usage))

    # Sort the list based on average CPU usage
    sorted_cores = sorted(core_usage_tuples, key=lambda x: x[1])

    # Get the first 'num_cores' indices (least active cores)
    least_active_cores = [index for index, _ in sorted_cores[:num_cores]]

    return least_active_cores


if __name__ == "__main__":

    experiment_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    args = parse_arguments()

    seed_everything(args.seed)

    logger.debug(f"Experiment started on: {experiment_datetime}")

    limit_cpu_cores(cpu_limit_cores=args.cpu_limit_cores)

    try:

        trial_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        logger.debug(
            f"Trial {nni.get_sequence_id() + 1} (# {nni.get_sequence_id()}, ID {nni.get_trial_id()}) "
            f"started on: {trial_datetime[:4]}-{trial_datetime[4:6]}-{trial_datetime[6:8]} "
            f"{trial_datetime[-6:-4]}:{trial_datetime[-4:-2]}:{trial_datetime[-2:]}\n"
        )
        n_tr = 200
        searchspace_filename = f"searchspace_{args.exp_name}"

        searchspace_path = f"{args.working_directory}/searchspaces/{searchspace_filename}.json"
        update_searchspace = SearchSpaceUpdater(
            {"filename": searchspace_path, "id": nni.get_experiment_id()}
        )
        if (nni.get_sequence_id() > 0) & (nni.get_sequence_id() % n_tr == 0):
            updater.update_searchspace(
                update_searchspace
            )  # it will use args.filename to update the search space

        params = nni.get_next_parameter()
        logger.debug(
            f"Parameters selected for trial {nni.get_sequence_id() + 1} "
            f"(# {nni.get_sequence_id()}, ID {nni.get_trial_id()}): {params}\n"
        )

        train_results, validation_results, test_accuracy = run_experiment(args, params)

        # Create directory and write to CSV
        path_report_results = Path(f"{args.working_directory}/results/reports/{args.exp_name}")
        path_report_results.mkdir(parents=True, exist_ok=True)
        csv_file_path = path_report_results / f"{nni.get_experiment_id()}.csv"

        save_test_results(csv_file_path=csv_file_path, test_accuracy=test_accuracy)

        path_best_model = Path(f"{args.working_directory}/results/best_model") / args.exp_name
        path_best_model.mkdir(parents=True, exist_ok=True)

        path_candidate_best_model = Path(
            os.path.join(os.environ["NNI_OUTPUT_DIR"], "tensorboard/checkpoints")
        )

        max_test_accuracy = get_max_accuracy_from_cvs(csv_file_path=csv_file_path)

        if test_accuracy * 100 >= max_test_accuracy and args.save_weights and len(train_results) == args.num_epochs:
            save_best_model(
                path_best_model=path_best_model,
                path_candidate_best_model=path_candidate_best_model
            )

        shutil.rmtree(path_candidate_best_model)

    except Exception as e:
        logger.exception(e)
        raise
