from matplotlib import pyplot as plt
import pandas as pd


def plot_result_losses_over_epoch(results_df: pd.DataFrame) -> plt.Figure:
    """
    Plot the losses over the epochs
    :param results_df: DataFrame containing the results
    :return: plt.Figure object with the errors shown over the epochs
    """
    fig, ax = plt.subplots(1, 3, figsize=(20, 10))

    ax = _plot_axis(
        ax=ax,
        ax_idx=0,
        results_df=results_df,
        x_col="epoch",
        y_train_col="train/cls_loss",
        y_val_col="val/cls_loss",
        title="Classification Loss",
        xlabel="Epoch",
        ylabel="Classification Loss",
    )
    ax = _plot_axis(
        ax=ax,
        ax_idx=1,
        results_df=results_df,
        x_col="epoch",
        y_train_col="train/box_loss",
        y_val_col="val/box_loss",
        title="Box Loss",
        xlabel="Epoch",
        ylabel="Box Loss",
    )
    ax = _plot_axis(
        ax=ax,
        ax_idx=2,
        results_df=results_df,
        x_col="epoch",
        y_train_col="train/dfl_loss",
        y_val_col="val/dfl_loss",
        title="DF Loss",
        xlabel="Epoch",
        ylabel="Distribution Focal Loss",
    )

    plt.tight_layout()
    plt.show()

    return fig


def _plot_axis(
    ax: plt.Axes,
    ax_idx: int,
    results_df: pd.DataFrame,
    x_col: str,
    y_train_col: str,
    y_val_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
) -> plt.Axes:
    """
    Plot the results of the training process
    :param ax:
    :param ax_idx: index of the axis
    :param results_df: dataframe containing the results
    :param x_col: column name of the x-axis
    :param y_train_col: column name of the y-axis for the training process
    :param y_val_col: column name of the y-axis for the validation process
    :param title: title of the plot
    :param xlabel: xlabel
    :param ylabel: ylabel
    :raises ValueError: if x_col, y_train_col, or y_val_col are not found in results_df
    :return: plt.Axes object with the plot
    """

    if (
        x_col not in results_df.columns
        or y_train_col not in results_df.columns
        or y_val_col not in results_df.columns
    ):
        raise ValueError(
            f"Column(s) {x_col}, {y_train_col}, or {y_val_col} not found in results_df"
        )

    x = results_df[x_col]
    y_train = results_df[y_train_col]
    y_val = results_df[y_val_col]

    ax[ax_idx].plot(x, y_train, label=f"Train {title}", color="blue")
    ax[ax_idx].plot(x, y_val, label=f"Validation {title}", color="orange")
    ax[ax_idx].set_title(title)
    ax[ax_idx].set_xlabel(xlabel)
    ax[ax_idx].set_ylabel(ylabel)
    ax[ax_idx].legend()
    ax[ax_idx].grid(True)

    return ax
