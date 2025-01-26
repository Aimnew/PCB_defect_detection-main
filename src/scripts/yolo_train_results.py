import shutil
from pathlib import Path

import pandas as pd
from pcb.visualizations.results import plot_result_losses_over_epoch
from pcb.utils import get_logger

logger = get_logger(__name__)


def eval_results(
    model_dir: str = "pcb_yolo11m_epochs_100_batch_-1",
    results_dir: Path = Path.cwd() / "results",
) -> None:
    """
    Evaluate the results of the YOLO11m model training
    :param model_dir: directory of the model
    :param results_dir: directory of the results
    :return:
    """
    # Check if training results exist
    results_model_dir = Path.cwd() / model_dir / "train"
    if not results_model_dir.exists():
        logger.error(f"Training results not found at {results_model_dir}")
        logger.info("Please run training first using 'poetry run yolo-train'")
        return

    # Create results directory if it doesn't exist
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Copying results to results directory.")
    try:
        shutil.copytree(src=results_model_dir, dst=results_dir, dirs_exist_ok=True)
    except FileNotFoundError as e:
        logger.error(f"Error copying results: {e}")
        return

    # Check if results file exists
    results_file = results_dir / "results.csv"
    if not results_file.exists():
        logger.error(f"Results file not found at {results_file}")
        return

    logger.info("Reading results.")
    try:
        results_df = pd.read_csv(results_file, index_col=0)
        results_df.columns = results_df.columns.str.strip()
        results_df = results_df.apply(pd.to_numeric, errors="coerce").dropna()
        results_df.reset_index(inplace=True)

        logger.info("Plotting results.")
        fig = plot_result_losses_over_epoch(results_df=results_df)

        logger.info("Saving results.")
        fig.savefig(results_dir / f"results_errors_{model_dir}.png")
    except Exception as e:
        logger.error(f"Error processing results: {e}")


def main():
    # ToDo: set directories in global config
    # ToDO: add click/argparse for arguments
    root_dir = Path.cwd().parent.resolve()
    results_dir = root_dir / "results"
    eval_results(
        model_dir="pcb_yolo11m_epochs_1_batch_-1",  # Updated to match your training epochs
        results_dir=results_dir
    )


if __name__ == "__main__":
    main()
