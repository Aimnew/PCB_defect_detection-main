from pathlib import Path

import torch.cuda
from ultralytics import YOLO
import shutil
import pandas as pd
from sahi.predict import predict

from pcb.utils import timer, get_logger, get_dataset_dir
from pcb.model.utils import read_yolo_labels_from_file, yolo_to_original_annot
from pcb.visualizations.annotations import visualize_annotations

logger = get_logger(__name__)


@timer
def inference(
    dest_results_dir: Path, 
    output_dir: Path,
    model_path: Path = Path("/Users/rey/Downloads/PCB_defect_detection-main/yolo11m.pt"),
    run_tiled_inference: bool = True
):
    """
    Run optimized inference on the test dataset
    """
    # Load the model
    if not model_path.exists():
        raise FileNotFoundError(f"YOLO11m model not found at {model_path}")
        
    model = YOLO(model_path)

    logger.info("Run inference on the test dataset")
    test_data_dir = output_dir / "images" / "val"

    model_device = "cuda" if torch.cuda.is_available() else "cpu"
    predict_dir = Path("runs/detect/predict")
    
    if run_tiled_inference:
        logger.info("Running tiled inference on the test dataset:")
        predict(
            model_type="yolo11m",
            model_path=str(model_path),
            model_device=model_device,
            model_confidence_threshold=0.25,
            source=test_data_dir,
            slice_height=320,
            slice_width=320,
            overlap_height_ratio=0.4,
            overlap_width_ratio=0.4,
        )
    else:
        logger.info("Running standard inference on the test dataset:")
        metrics = model(
            source=test_data_dir,
            imgsz=640,
            conf=0.25,
            iou=0.45,
            save=True,
            save_txt=True,
            save_conf=True,
            augment=True,
            project=predict_dir.parent,
            name=predict_dir.name
        )
    logger.info("Inference completed.")

    return metrics, predict_dir


def main():
    # Get dataset directory
    dataset_dir = get_dataset_dir()
    dest_results_dir = Path.cwd().parent.resolve() / "results"
    output_dir = dataset_dir / "output"

    # LOAD ANNOTATIONS
    annot_df = pd.read_parquet(dataset_dir / "annotation.parquet")

    metric, predict_dir = inference(
        dest_results_dir=dest_results_dir, 
        output_dir=output_dir,
        model_path=Path("/Users/rey/Downloads/PCB_defect_detection-main/yolo11m.pt")
    )
    logger.debug(f"Metrics: {metric}")

    # Copy results
    dest_predict_dir = dest_results_dir / "predict"
    if predict_dir.exists():
        shutil.copytree(predict_dir, dest_predict_dir, dirs_exist_ok=True)
    else:
        logger.error(f"Prediction directory not found at {predict_dir}")
        return

    # Analysis of specific test case
    test_name = "12_spurious_copper_05"
    labels_dir = predict_dir / "labels"
    if not labels_dir.exists():
        logger.warning("No detections found in validation set")
        return
        
    file_path = labels_dir / f"{test_name}.txt"
    if not file_path.exists():
        logger.warning(f"No detection file found for {test_name}")
        return
        
    yolo_labels = read_yolo_labels_from_file(file_path)

    classes = [
        "missing_hole",
        "mouse_bite", 
        "open_circuit",
        "short",
        "spur",
        "spurious_copper"
    ]

    pred_annot_df = yolo_to_original_annot(
        image_name=f"{test_name}.jpg",
        yolo_labels=yolo_labels,
        annot_df=annot_df,
        classes=classes,
    )

    visualize_annotations(
        image_name=f"{test_name}.jpg",
        images_dir=str(dataset_dir / "images"),
        annot_df=pred_annot_df,
        is_subfolder=True,
    )

    visualize_annotations(
        image_name=f"{test_name}.jpg", 
        images_dir=str(dataset_dir / "images"),
        annot_df=annot_df,
        is_subfolder=True,
    )


if __name__ == "__main__":
    main()
