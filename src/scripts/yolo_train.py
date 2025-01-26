from typing import Literal
from ultralytics import YOLO
from pathlib import Path
from pcb.utils import get_logger, timer, get_dataset_dir
import torch


logger = get_logger(__name__)


def _is_mps_available() -> bool:
    """
    Check if MPS is available
    :return: boolean whether MPS is available
    """
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            logger.warning(
                "MPS not available because the current PyTorch install was not "
                "built with MPS enabled."
            )
        else:
            logger.warning(
                "MPS not available because the current MacOS version is not 12.3+ "
                "and/or you do not have an MPS-enabled device on this machine."
            )

        return False

    return True


@timer
def train(
    data_path: Path = Path.cwd() / "data.yaml",
    model_path: Path = Path("/Users/rey/Downloads/PCB_defect_detection-main/yolo11m.pt"),
    epochs: int = 100,
    batch: int = 16,
    imgsz: int = 640,
    save_period: int = 5,
    verbose: bool = True,
    mixup: float = 0.3,
    device: Literal["cpu", "mpg", 0] | None = None,
    mosaic: float = 1.0,
):
    """
    Train a YOLO11m model with improved parameters
    """
    logger.info("Start training.")
    project = f"pcb_yolo11m_epochs_{epochs}_batch_{batch}"

    if not model_path.exists():
        raise FileNotFoundError(f"YOLO11m model not found at {model_path}")
        
    if device == "mps":
        # ToDo: current implementation only considers MPS for MacOS or CPUs and not for GPUs
        if not _is_mps_available():
            logger.warning("MPS is not available. Switching to CPU.")
            logger.info("If GPUs are available, please specify the device as 'cuda'.")
            device = "cpu"
    if device == "cuda":
        if not torch.cuda.is_available():
            logger.warning("GPU is not available. Switching to CPU.")
            device = "cpu"

    if isinstance(batch, float):
        if device != "cuda":
            logger.warning(
                "Batch size must be an integer when not running on GPU. Switching to default batch size of "
                "16."
            )

    logger.info(f"Storing data in project: {project}")
    model_yolo = YOLO(str(model_path))
    result = model_yolo.train(
        data=str(data_path),
        epochs=epochs,
        batch=batch,
        lr0=0.001,
        lrf=0.0001,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        imgsz=imgsz,
        save_period=save_period,
        verbose=verbose,
        project=project,
        mixup=mixup,
        device=device,
        mosaic=mosaic,
        augment=True,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0,
        flipud=0.5,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        copy_paste=0.1,
    )

    logger.info("Training completed.")
    return result


def main():
    # Get dataset directory
    dataset_dir = get_dataset_dir()
    output_dir = dataset_dir / "output"

    # Create data.yaml
    all_data_yaml = f"""
        path: {output_dir}
        train: images/train
        val: images/val

        names:
            0: missing_hole
            1: mouse_bite
            2: open_circuit
            3: short
            4: spur
            5: spurious_copper
        """

    data_path = Path.cwd().parent.resolve() / "data.yaml"

    with open(data_path, "w") as f:
        f.write(all_data_yaml)

    train(
        data_path=data_path,
        model_path=Path("/Users/rey/Downloads/PCB_defect_detection-main/yolo11m.pt"),
        epochs=1,
        batch=-1,
        imgsz=640,
        save_period=5,
        verbose=True,
        mixup=0.3,
        device="cuda",
        mosaic=1.0,
    )


if __name__ == "__main__":
    main()
