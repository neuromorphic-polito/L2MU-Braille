import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from network_engine import NetworkEngine
from lightning.pytorch import Trainer, seed_everything
from nni_suite.utils import *
import random
from nni_experiments_training import (
    load_model,
    load_data,
    configure_trainer,
    get_checkpoint_file_path,
    process_train_val_results,
    process_test_result,
    save_best_model,
    limit_cpu_cores
)
import shutil
import pandas as pd
import torch
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
import datetime
import logging
import argparse
from pathlib import Path

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

torch.set_float32_matmul_precision("high")


def run_experiment(args, params):
    data_module = load_data(
        dataset_path=args.dataset_path, batch_size=int(params["batch_size"]), split=args.split
    )

    model = load_model(
        num_inputs=data_module.num_inputs,
        num_outputs=data_module.num_outputs,
        params=params,
        architecture=args.architecture,
    )

    tensorboard_log_dir = Path(f"{args.working_directory}/post_nni_opt/tensorboard/{args.exp_name}/split_{args.split}")

    trainer, callback_training_history = configure_trainer(
        num_epochs=args.num_epochs,
        nni_report=False,
        tensorboard_log_dir=tensorboard_log_dir,
        gpu_index=args.gpu_index
    )

    ckpt_path = get_checkpoint_file_path(tensorboard_log_dir=tensorboard_log_dir)

    trainer.fit(model=model, datamodule=data_module, ckpt_path=ckpt_path)

    training_results = callback_training_history.training_results
    validation_results = callback_training_history.validation_results

    training_hist = np.array(training_results)
    validation_hist = np.array(validation_results)

    process_train_val_results(
        training_hist=training_hist, validation_hist=validation_hist, trainer=trainer
    )

    test_accuracy = process_test_result(
        trainer=trainer, model=model, data_module=data_module
    )

    return training_results, validation_results, test_accuracy


def test_run(args, params, path_model):
    model = NetworkEngine.load_from_checkpoint(path_model, strict=False)
    data_module = load_data(dataset_path=args.dataset_path, batch_size=int(params['batch_size']))

    logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)

    trainer = Trainer(
        accelerator="gpu",
        devices=[0],
        num_sanity_val_steps=0,
        enable_progress_bar=False,
        enable_checkpointing=False,
        logger=False
    )
    test_data = trainer.test(
        model=model, datamodule=data_module, verbose=False
    )[0]

    return test_data["test_accuracy"]


def generate_confusion_matrix(dataset_path, params, path_model, path_plot, experiment_id, selected_id,
                              experiment_datetime, save_fig):
    spiking_network = NetworkEngine.load_from_checkpoint(path_model, strict=False)
    data_module = load_data(dataset_path=dataset_path, batch_size=int(params["batch_size"]))

    logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)

    trainer = Trainer(
        accelerator="gpu",
        devices=[0],
        num_sanity_val_steps=0,
        enable_progress_bar=False,
        enable_checkpointing=False,
        logger=False
    )

    preds = trainer.predict(
        model=spiking_network, datamodule=data_module, return_predictions=True
    )

    preds = np.concatenate(preds).ravel().tolist()

    trues = (
        data_module.test_dataset[:][1].cpu().tolist()
    )

    labels = data_module.label_names

    cm = confusion_matrix(trues, preds, normalize="true")
    cm_df = pd.DataFrame(cm, index=[ii for ii in labels], columns=[jj for jj in labels])
    plt.figure("cm", figsize=(12, 9))
    sn.heatmap(cm_df, annot=True, fmt=".2f", cbar=False, square=False, cmap="YlOrBr",
               annot_kws={"size": 20, "fontfamily": "serif"})
    plt.xlabel("\nPredicted", fontsize=21, fontweight='medium')
    plt.ylabel("True\n", fontsize=21, fontweight='medium')
    plt.xticks(rotation=0, fontsize=18, fontweight='medium')
    plt.yticks(rotation=0, fontsize=18, fontweight='medium')
    plt.tight_layout()

    if save_fig:
        path_to_save_fig = path_plot / f'cm_{experiment_id}_{selected_id}_{experiment_datetime}'

        plt.savefig(f'{path_to_save_fig}.png', dpi=300)
        plt.savefig(f'{path_to_save_fig}.pdf', dpi=300)
        plt.savefig(f'{path_to_save_fig}.eps', dpi=300)
    else:
        plt.show()
    plt.close()


def parse_arguments():
    parser = argparse.ArgumentParser()

    # Name for the experiment
    parser.add_argument(
        "-exp_name",
        type=str,
        default="braille_full_exploration_l2mu",
        help="Name for the starting experiment.",
    )
    # ID of the NNI experiment to refer to
    parser.add_argument(
        "-experiment_id",
        type=str,
        default="0id1ys96",
        help="ID of the NNI experiment whose results are to be used.",
    )

    parser.add_argument(
        "-trial_id",
        type=str,
        default="vWt9b",
        help="Trial id of the NNI experiment.",
    )

    parser.add_argument(
        "-onnx_export",
        type=bool,
        default=False,
        help="Set to True if you want to export the model into onnx",
    )

    # Filepath for Braille data
    parser.add_argument(
        "-dataset_path",
        type=str,
        default="../data/braille_full_splitted",
        help="Path for the dataset to be loaded.",
    )

    parser.add_argument(
        "-working_directory",
        type=str,
        default="../model_insights",
        help="Path for the working directory.",
    )

    parser.add_argument(
        "-limit_cpu_cores",
        type=int,
        default=4,
        help="Number of CPU cores to use",
    )

    # Number or repetitions for training statistics
    parser.add_argument(
        "-repetitions",
        type=int,
        default=10,  # default: 10
        help="Number of trainings to be performed for statistical evaluation.",
    )

    # Number of epochs
    parser.add_argument(
        "-num_epochs", type=int, default=300, help="Number of training epochs."
    )

    # Save figures
    parser.add_argument(
        "-save_fig",
        type=bool,
        default=True,
        help="Save or not the plots produced during training and test.",
    )
    # Save the weights from optimal results
    parser.add_argument(
        "-save_weights",
        type=bool,
        default=True,
        help="Weights can be saved to be loaded after training and used for test.",
    )

    # Set default seed
    parser.add_argument(
        "-default_seed",
        type=int,
        default=42,
        help="Default seed used during nni experiments",
    )

    parser.add_argument(
        "-retrain_model",
        type=bool,
        default=True,
        help="Set if you want to extract statistics/learning curves and confusion matrix",
    )

    parser.add_argument(
        "-architecture",
        type=str,
        choices=["L2MU"],
        default="L2MU",
        help="The network architecture.",
    )

    parser.add_argument(
        "-nni_metrics",
        type=bool,
        default=True,
        help="Take metrics and params from nni best model",
    )

    parser.add_argument(
        "-gpu_index",
        type=int,
        default=2,
        help="gpu index",
    )

    return parser.parse_args()


def generate_metric_plot(
        metric_train_data,
        metric_val_data,
        path_figures,
        experiment_datetime,
        experiment_id,
        selected_id,
        metric,
        statistics=True,
        save_fig=False
):
    """
    Make plots for metric from training and validation
    Compute mean, median and std. dev.
    """

    train_data = metric_train_data
    val_data = metric_val_data

    if statistics:
        metric_median_train = np.median(metric_train_data, axis=0)
        metric_median_val = np.median(metric_val_data, axis=0)

        train_data = metric_median_train
        val_data = metric_median_val

    plt.figure()

    y_label = ""
    pos_legend = ""
    scale_factor = 1
    y_lim = None

    match metric:
        case "accuracy":
            y_label = f'{metric.capitalize()} (%)'
            pos_legend = "lower right"
            scale_factor = 100
            y_lim = (0, 100)

        case "loss":
            y_label = f'{metric.capitalize()}'
            pos_legend = "center right"
            scale_factor = 1
            y_lim = (0, None)

    plt.plot(
        range(1, len(train_data) + 1),
        scale_factor * np.array(train_data),
        color="orangered",
    )
    plt.plot(
        range(1, len(val_data) + 1),
        scale_factor * np.array(val_data),
        color="tab:green",
    )

    if statistics:
        std_train_data = np.std(train_data, axis=0)
        std_val_data = np.std(val_data, axis=0)

        plt.fill_between(
            range(1, len(train_data) + 1),
            scale_factor * (train_data + std_train_data),
            scale_factor * (train_data - std_train_data),
            color="sandybrown",
        )
        plt.fill_between(
            range(1, len(val_data) + 1),
            scale_factor * (val_data + std_val_data),
            scale_factor * (val_data - std_val_data),
            color="lightgreen",
        )

    plt.xlabel("Epoch")
    plt.ylabel(y_label)
    plt.ylim(y_lim)
    plt.legend(["Training", "Validation"], loc=pos_legend)
    plt.tight_layout()

    if save_fig:

        plt.savefig(
            path_figures
            / f"{metric}_{experiment_id}_{selected_id}_{experiment_datetime}_stats.pdf",
            dpi=300,
        )
        plt.savefig(
            path_figures
            / f"{metric}_{experiment_id}_{selected_id}_{experiment_datetime}_stats.png",
            dpi=300,
        )
        plt.savefig(
            path_figures
            / f"{metric}_{experiment_id}_{selected_id}_{experiment_datetime}_stats.eps",
            dpi=300,
        )
    else:
        plt.show()

    plt.close()


def parameter_statistic(path_model):
    model_ckpt = NetworkEngine.load_from_checkpoint(path_model, strict=False)

    torch.save(model_ckpt.state_dict(), "temp.ckpt")
    print('Model Size (MB):', os.path.getsize("temp.ckpt") / 1e6)
    os.remove('temp.ckpt')

    size_model = 0
    for param in model_ckpt.parameters():
        if param.data.is_floating_point():
            size_model += param.numel() * torch.finfo(param.data.dtype).bits
        else:
            size_model += param.numel() * torch.iinfo(param.data.dtype).bits

    for buffer in model_ckpt.buffers():
        if buffer.data.is_floating_point():
            size_model += buffer.numel() * torch.finfo(buffer.data.dtype).bits
        else:
            size_model += buffer.numel() * torch.iinfo(buffer.data.dtype).bits

    print(f"Model Size: {size_model} / bit | {size_model / 8e6} / MB")

    trainable_params = 0

    for parameter in model_ckpt.parameters():
        if parameter.requires_grad:
            trainable_params += parameter.numel()

    tot_params = 0
    for name, param in model_ckpt.state_dict().items():
        tot_params += param.numel()

    print(f"Total Params: {tot_params}")
    print(f"Num. Trainable Params: {trainable_params}")


def export_to_onnx(path_model, args):
    ckpt = NetworkEngine.load_from_checkpoint(path_model, map_location='cpu')
    model_name_onnx = f"{args.architecture}.onnx" if args.architecture not in ['L2MU',
                                                                               'SRNN'] else f"{args.architecture}_{args.encoder}_{args.neuron_type}.onnx"
    filepath = path_model.parents[0] / model_name_onnx
    if not filepath.exists():
        print('Exporting...')
        input_sample = torch.randn((40, 1, 6))
        ckpt.to_onnx(filepath, input_sample, export_params=True)
    print('Exported Model Size (MB):', os.path.getsize(filepath) / 1e6)


if __name__ == "__main__":

    args = parse_arguments()

    log_path = Path(f"{args.working_directory}/post_nni_opt/logs")
    log_path.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_path / f'{args.exp_name}.log',
        filemode="a",
        format="%(asctime)s %(name)s %(message)s",
        datefmt="%Y-%m-%d_%H:%M:%S",
    )

    experiment_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    logger.debug(f"Experiment started on: {experiment_datetime}")

    limit_cpu_cores(args.limit_cpu_cores)

    params = None
    best_test_id = None
    best_metric_from_hpo = None

    if args.nni_metrics:

        logger.debug(f"Get best model from the NNI default path")
        NNI_metrics = "test accuracy"

        trial_id = args.trial_id if len(args.trial_id) != 0 else None

        # Get the best NNI result and the corresponding hyperparameters
        best_test_id, best_NNI_metrics, params = retrieve_nni_results(exp_name=args.exp_name, exp_id=args.experiment_id,
                                                                      metrics=NNI_metrics,
                                                                      trial_id=trial_id,
                                                                      nni_default_path='../nni-experiments',
                                                                      working_directory='../model_insights'
                                                                      )

        best_metric_from_hpo = best_NNI_metrics

        logger.debug(f"Experiment name: {args.exp_name}")
        logger.debug(f"Experiment ID (NNI): {args.experiment_id}")
        logger.debug(f"\tBest test ID (NNI): {best_test_id}")
        logger.debug(f"\tBest NNI metrics ({NNI_metrics}): {best_NNI_metrics}")

    else:
        logger.debug(f"Get best model from the best model path folder")
        path_params = Path(f"{args.working_directory}/best_parameters/{args.exp_name}")
        file_path = next(path_params.iterdir(), None)
        if file_path and file_path.is_file():
            with open(file_path) as f:
                params = json.load(f)

        path_best_model_from_hpo = Path(f"{args.working_directory}/results/best_model/{args.exp_name}/best.ckpt")
        best_metric_from_hpo = test_run(args, params, path_best_model_from_hpo) * 100

        logger.debug(f"Experiment name: {args.exp_name}")
        logger.debug(f"\tBest NNI metrics (test_accuracy): {best_metric_from_hpo}")

    if args.repetitions > 0:

        best_test_accuracy = -1
        best_seed = -1

        path_best_model = (
                Path(f"{args.working_directory}/best_model_from_stats")
                / args.exp_name
        )
        path_best_model.mkdir(parents=True, exist_ok=True)

        logger.debug("Retrain model from scratch to extract statistics from multiple train-val splits")

        start_exp_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        print(
            f"*** training (with validation) statistics started ({start_exp_datetime}) ***"
        )
        logger.debug(
            f"### Training statistics with {args.repetitions} repetitions started ({start_exp_datetime}). ###\n"
        )

        splits_list = list(range(args.repetitions))
        seed_everything(args.default_seed)
        acc_test_list = []

        for rpt, split in enumerate(splits_list):

            logger.debug(f"REPETITION {rpt + 1}/{len(splits_list)}")



            # Train the network with validation and test
            args.split = split
            training_results, validation_results, test_accuracy = run_experiment(
                    args, params
            )

            path_candidate_best_model = Path(
                f"{args.working_directory}/post_nni_opt/tensorboard/{args.exp_name}/split_{split}"
            )

            # Save layers providing the best median test accuracy
            if test_accuracy > best_test_accuracy:
                best_test_accuracy = test_accuracy
                best_split = split
                if args.save_weights:
                    save_best_model(
                        path_best_model=path_best_model,
                        path_candidate_best_model=path_candidate_best_model,
                    )
            acc_test_list.append(test_accuracy)

            end_exp_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            logger.debug(
                f"\trepetition {rpt + 1}/{len(splits_list)} done ({end_exp_datetime}) --> "
                f"test accuracy: {np.round(test_accuracy * 100, 2)}% with split {split}"
            )
            print(f"\trepetition {rpt + 1}/{len(splits_list)} done ({end_exp_datetime}) --> "
                  f"test accuracy: {np.round(test_accuracy * 100, 2)}% with split {split}")

            end_exp_statistics = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            logger.debug(
                f'### Training statistics done ({end_exp_statistics}). ###\n'
            )
            logger.debug(f'### Calculate statistics on test accuracies. ###\n')
            median = np.median(acc_test_list)
            logger.debug(f"Median: {median}")
            mean = np.mean(acc_test_list)
            logger.debug(f"Mean: {mean}")
            std = np.std(acc_test_list)
            logger.debug(f"Std: {std}")
            print("*** training (with validation) statistics done ***")
            print(f"Median, Mean, Std: {median}, {mean}, {std}")

    else:

        logger.debug("Retrain model from scratch to extract learning curves from default seed")
        seed_everything(args.default_seed)
        args.split = 0
        training_results, validation_results, test_accuracy = run_experiment(args, params)

    if args.repetitions == 0 and args.retrain_model == False:

        logger.debug(f"Use nni best model to calculate test accuracy and generate confusion matrix")

        path_model = Path(f'{args.working_directory}/results/best_model/{args.exp_name}/best.ckpt')
        test_accuracy = test_run(args=args, params=params, path_model=path_model)
        print(f"Test accuracy is {test_accuracy * 100}")

        logger.debug(
            f'Test accuracy: {np.round(test_accuracy * 100, 4)}%'
        )

        path_figure = Path(
            f"{args.working_directory}/plots/confusion_matrices/{args.exp_name}"
        )
        path_figure.mkdir(parents=True, exist_ok=True)

        generate_confusion_matrix(params=params, dataset_path=args.dataset_path, path_model=path_model,
                                  path_plot=path_figure, experiment_id=args.experiment_id, selected_id=best_test_id,
                                  experiment_datetime=experiment_datetime, save_fig=args.save_fig)
        parameter_statistic(path_model)
        if args.onnx_export:
            export_to_onnx(path_model, args)

    conclusion_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    logger.debug(f"Experiment concluded on: {conclusion_datetime}")

    logger.debug(
        "------------------------------------------------------------------------------------\n\n"
    )
