from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import scienceplots


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
    Generate aesthetically pleasing and grayscale-compatible plots for training and validation metrics
    with distinct error regions using transparency differences.
    """

    plt.style.use(['science','ieee'])

    train_data = metric_train_data
    val_data = metric_val_data

    if statistics:
        metric_median_train = np.median(metric_train_data, axis=0)
        metric_median_val = np.median(metric_val_data, axis=0)

        train_data = metric_median_train
        val_data = metric_median_val

    plt.figure(figsize=(6, 4))

    y_label = ""
    pos_legend = ""
    scale_factor = 1
    y_lim = None

    match metric:
        case "accuracy":
            y_label = f'{metric.capitalize()} (%)'
            pos_legend = "lower right"
            scale_factor = 100
            y_lim = (0, 105)

        case "loss":
            y_label = f'{metric.capitalize()}'
            pos_legend = "upper right"
            scale_factor = 1
            y_lim = (0, 12)

    # Plot training data: solid red line
    plt.plot(
        range(1, len(train_data) + 1),
        scale_factor * np.array(train_data),
        color="#CE8147",
        linestyle='-',  # Solid line for training
        linewidth=2,    # Standard thickness for training
        label="Training"
    )

    # Plot validation data: dashed green line
    plt.plot(
        range(1, len(val_data) + 1),
        scale_factor * np.array(val_data),
        color="#1D8A99",
        linestyle='--',  # Dashed line for validation
        linewidth=2,     # Standard thickness for validation
        label="Validation"
    )

    if statistics:
        std_train_data = np.std(metric_train_data, axis=0)
        std_val_data = np.std(metric_val_data, axis=0)

        # Error bars for training data (more opaque fill)
        plt.fill_between(
            range(1, len(train_data) + 1),
            scale_factor * np.clip(train_data + std_train_data, 0, 100/scale_factor),
            scale_factor * np.clip(train_data - std_train_data, 0, 100/scale_factor),
            color="#CE8147",  # Opaque fill for training error region
            hatch='\\',
            zorder=1,
            alpha=0.3,
        )

        # Error bars for validation data (lighter fill for contrast)
        plt.fill_between(
            range(1, len(val_data) + 1),
            scale_factor * np.clip(val_data + std_val_data, 0, 100/scale_factor),
            scale_factor * np.clip(val_data - std_val_data, 0, 100/scale_factor),
            color="#1D8A99",
            hatch='.',
            alpha=0.3,  # Lighter fill for validation error region
            zorder=2
        )

    plt.xlabel("Epoch")
    plt.ylabel(y_label)
    plt.ylim(y_lim)
    plt.xlim(-5, 305)


    # Simple, clean legend
    plt.legend(loc=pos_legend)

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
        # plt.savefig(
        #     path_figures
        #     / f"{metric}_{experiment_id}_{selected_id}_{experiment_datetime}_stats.eps",
        #     dpi=300,
        # )
    else:
        plt.show()

    plt.close()


if __name__ == "__main__":

    # Specify the folder path
    experiment_name = 'braille_full_exploration_l2mu'
    folder_path = Path(f'/home/leto/nice_2025/model_insights/post_nni_opt/tensorboard/{experiment_name}')

    # Find all files that end with '.ckpt'
    ckpt_files = list(folder_path.glob('**/*last.ckpt'))

    loss_train_list = []
    acc_train_list = []
    loss_val_list = []
    acc_val_list = []

    # Print the list of found files
    for file in ckpt_files:
        x = torch.load(file, weights_only=False)
        training_results = x['callbacks']['ReportMetrics']['train_results']
        validation_results = x['callbacks']['ReportMetrics']['validation_results']
        #print(len(validation_results))

        loss_hist = [
            np.array(training_results)[:, 0],
            np.array(validation_results)[:, 0],
        ]
        acc_hist = [
            np.array(training_results)[:, 1],
            np.array(validation_results)[:, 1],
        ]

        loss_train_list.append(loss_hist[0])
        acc_train_list.append(acc_hist[0])
        loss_val_list.append(loss_hist[1])
        acc_val_list.append(acc_hist[1])

    path_figures = Path(f'../model_insights/plots/learning_curves/{experiment_name}')
    path_figures.mkdir(parents=True, exist_ok=True)

    generate_metric_plot(
        metric_train_data=acc_train_list,
        metric_val_data=acc_val_list,
        path_figures=path_figures,
        experiment_id=121,
        experiment_datetime=41,
        selected_id=21,
        metric='accuracy',
        statistics=True,
        save_fig=True
    )

    generate_metric_plot(
        metric_train_data=loss_train_list,
        metric_val_data=loss_val_list,
        path_figures=path_figures,
        experiment_id=121,
        experiment_datetime=41,
        selected_id=21,
        metric='loss',
        statistics=True,
        save_fig=True
    )
