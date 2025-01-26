import logging
from typing import Literal
import functools
import time
import os
import requests
from pathlib import Path
import shutil

_LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(funcName)s() --  %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT, datefmt=_DATE_FORMAT)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the configuration defined in this file.

    :param name: Name of the logger (usually __name__ / for developing __file__).
    :return:     Logger with the specified name and configuration.
    """

    if "src/" in name:
        name = name.split("src/")[-1]

    logger = logging.getLogger(name)

    return logger


def timer(func: callable) -> callable:
    """
    Decorator to measure the time of a function
    :param func: any exectuable function
    :return:
    """
    logger = get_logger(__name__)

    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        tic = time.perf_counter()
        value = func(*args, **kwargs)
        toc = time.perf_counter()
        elapsed_time = toc - tic
        logger.info(f"Elapsed time: {elapsed_time:0.4f} seconds")
        return value

    return wrapper_timer


def get_subfolder(
    image_name: str,
) -> Literal[
    "Missing_hole", "Mouse_bite", "Open_circuit", "Short", "Spur", "Spurious_copper"
]:
    """
    Get the subfolder name from the image name
    :param image_name: file name of the image
    :return: class name
    """
    if "missing" in image_name.split("_"):
        return "Missing_hole"
    if "mouse" in image_name.split("_"):
        return "Mouse_bite"
    if "open" in image_name.split("_"):
        return "Open_circuit"
    if "short" in image_name.split("_"):
        return "Short"
    if "spur" in image_name.split("_"):
        return "Spur"
    if "spurious" in image_name.split("_"):
        return "Spurious_copper"


def download_dataset(target_dir: Path) -> None:
    """
    Download PCB_DATASET if not present
    """
    logger = get_logger(__name__)
    
    # URL for PCB dataset download
    dataset_url = "https://universe.roboflow.com/ds/PCB_DATASET?key=your_key"  # Replace with actual URL
    
    logger.info(f"Downloading PCB_DATASET to {target_dir}")
    try:
        response = requests.get(dataset_url, stream=True)
        zip_path = target_dir / "PCB_DATASET.zip"
        
        with open(zip_path, 'wb') as f:
            shutil.copyfileobj(response.raw, f)
            
        # Extract dataset
        shutil.unpack_archive(zip_path, target_dir)
        os.remove(zip_path)
        
        logger.info("Dataset downloaded and extracted successfully")
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        raise

def get_dataset_dir() -> Path:
    """
    Get PCB_DATASET directory, download if not present
    """
    logger = get_logger(__name__)
    
    # Check custom location first
    custom_path = Path("/Users/rey/Downloads/PCB_defect_detection-main/PCB_DATASET")
    if custom_path.exists():
        logger.info(f"Using PCB_DATASET from {custom_path}")
        return custom_path
        
    # Otherwise use default location
    default_path = Path.cwd().parent.resolve() / "PCB_DATASET"
    if not default_path.exists():
        logger.warning(f"PCB_DATASET not found at {default_path}")
        logger.info("Would you like to download the dataset? (y/n)")
        response = input().lower()
        if response == 'y':
            download_dataset(default_path.parent)
        else:
            raise FileNotFoundError("PCB_DATASET is required but not found")
            
    return default_path
