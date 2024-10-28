import argparse
import json
import logging
import sys
import torch
from pathlib import Path
from nni.experiment import Experiment, ExperimentConfig, LocalConfig, AlgorithmConfig
torch.set_float32_matmul_precision('high')


# Configure logging
def configure_logging():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    return logger


# Parse command line arguments
def parse_arguments():
    parser = argparse.ArgumentParser()

    # Add arguments to the parser
    parser.add_argument(
        "-exp_name_base",
        type=str,
        default="wisdm_exploration",
        help="Name for the starting experiment.",
    )
    parser.add_argument(
        "-exp_trials",
        type=int,
        default=10000,
        help="Number of trials for the starting experiment.",
    )
    parser.add_argument(
        "-exp_time",
        type=str,
        default="100d",
        help="Maximum duration of the starting experiment.",
    )
    parser.add_argument(
        "-exp_gpu_number",
        type=int,
        default=1,
        help="How many GPUs to use for the starting experiment.",
    )
    parser.add_argument(
        "-gpu_index",
        type=int,
        default=0,
        help="GPU index to be used for the experiment.",
    )
    parser.add_argument(
        "-exp_concurrency",
        type=int,
        default=1,
        help="Concurrency for the starting experiment.",
    )
    parser.add_argument(
        "-max_per_gpu", type=int, default=5, help="Maximum number of trials per GPU."
    )
    parser.add_argument(
        "-port", type=int, default=8080, help="Port number for the starting experiment."
    )
    parser.add_argument(
        "-cpu_limit_cores", type=int, default=-1, help="Limited number of cpu cores to use."
    )
    parser.add_argument(
        "-save_weights",
        type=bool,
        default=True,
        help="Whether to save weights after training.",
    )
    parser.add_argument(
        "-seed", type=int, default=42, help="Set seed for reproducibility."
    )
    parser.add_argument(
        "-dataset_path",
        type=str,
        default="../data/wisdm-dataset/full_dataset.npz",
        help="Path to the dataset.",
    )

    parser.add_argument(
        "-num_epochs",
        type=int,
        default=300,
        help="Number of epochs the model should be trained.",
    )

    parser.add_argument(
        "-architecture",
        type=str,
        choices=["L2MU"],
        default="L2MU",
        help="The network architecture.",
    )
    parser.add_argument(
        "-script",
        type=str,
        default="nni_experiments_training.py",
        help="Path of the training script.",
    )

    parser.add_argument("-working_directory",
                        type=str,
                        default="../model_insights",
                        help="Path of the working directory.", )

    return parser.parse_args()


# Create and configure the NNI experiment
def create_experiment(args):
    # Construct search space and experiment paths
    network_search_space_path = (
        f"../searchspace/{args.architecture}/searchspace.json"
    )
    searchspace_filename = (
        f"searchspace_{args.exp_name_base}_{args.architecture.lower()}"
    )
    experiment_name = f"{args.exp_name_base}_{args.architecture.lower()}"

    # Load and save search space
    with open(network_search_space_path, "r") as f:
        searchspace = json.load(f)

    searchspace_path = Path(f"{args.working_directory}/searchspaces")
    searchspace_path.mkdir(parents=True, exist_ok=True)
    searchspace_filepath = searchspace_path / f'{searchspace_filename}.json'

    # searchspace_path = f"{args.working_directory}/searchspaces/{searchspace_filename}.json"
    with open(searchspace_filepath, "w") as write_searchspace:
        json.dump(searchspace, write_searchspace)

    # Configure the experiment
    config = ExperimentConfig(
        experiment_name=experiment_name,
        experiment_working_directory="../nni-experiments/{}".format(
            f"{experiment_name}"
        ),
        trial_command=f"python3 {args.script} -exp_name {experiment_name} -gpu_index {args.gpu_index}"
                      f" -cpu_limit_cores {args.cpu_limit_cores} -save_weights {args.save_weights} -seed {args.seed} -dataset_path {args.dataset_path}"
                      f" -architecture {args.architecture}  -working_directory {args.working_directory}"
                      f" -num_epochs {args.num_epochs}",
        trial_code_directory=Path(__file__).parent,
        search_space=searchspace,
        tuner=AlgorithmConfig(name="Anneal", class_args={"optimize_mode": "maximize"}),
        assessor=AlgorithmConfig(
            name="Medianstop",
            class_args=({"optimize_mode": "maximize", "start_step": 10}),
        ),
        tuner_gpu_indices=args.gpu_index,
        max_trial_number=args.exp_trials,
        max_experiment_duration=args.exp_time,
        trial_concurrency=args.exp_concurrency,
        training_service=LocalConfig(trial_gpu_number=args.exp_gpu_number,
                                     max_trial_number_per_gpu=args.max_per_gpu,
                                     use_active_gpu=True,
                                     gpu_indices=args.gpu_index
                                     ),
    )

    return Experiment(config)


# Run the experiment
def run_experiment(experiment, port):
    experiment.run(port)

    # Wait for user input to stop
    input("Press any key to stop the experiment.")
    experiment.stop()


# Main function
def main():
    logger = configure_logging()
    args = parse_arguments()
    experiment = create_experiment(args)
    run_experiment(experiment, args.port)


if __name__ == "__main__":
    main()
