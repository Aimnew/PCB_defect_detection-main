import time
from pathlib import Path
import pandas as pd
from ultralytics import YOLO

# %%
# GLOBAL SETTINGS (also used in scripts/yolo_preprocess.py
ksplit: int = 3
dataset_dir = Path.cwd().parent.resolve() / "PCB_DATASET"
output_dir = dataset_dir / "output"
output_dir_kfold = Path(output_dir / f"{ksplit}fold_crossval")

folds_df = pd.read_csv(output_dir_kfold / "kfold_datasplit.csv", index_col=0)

# %%
# Create directories and dataset YAML files
ds_yamls = []
save_path = Path(output_dir / f"{ksplit}fold_crossval")

for split in folds_df.columns:
    # Create directories
    split_dir = save_path / split
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / "train" / "images").mkdir(parents=True, exist_ok=True)
    (split_dir / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (split_dir / "val" / "images").mkdir(parents=True, exist_ok=True)
    (split_dir / "val" / "labels").mkdir(parents=True, exist_ok=True)

    # Create dataset YAML files
    dataset_yaml = split_dir / f"{split}_dataset.yaml"
    ds_yamls.append(dataset_yaml)

# MODEL TRAINING
## SETTINGS
# %%
model_name = "yolo11m"
model_path = Path("/Users/rey/Downloads/PCB_defect_detection-main/yolo11m.pt")

if not model_path.exists():
    raise FileNotFoundError(f"YOLO11m model not found at {model_path}")
    
model = YOLO(str(model_path))

### Additional arguments here
batch = 16  # if batch=-1 it uses 60% of GPU memory, of set to float <1: uses % of GPU memory, if int, uses batch
#   size - ONLY WHEN RUN ON GPU
epochs = 10
imgsz = 640
save_period = 1  # Save model every `save_period` epochs
verbose = True
# box=6.0       # Weight of the box loss component in the loss function, default 7.5, influencing how much emphasis
#                       is placed on accurately predicting bounding box coordinates
# cls = 2.0     # Weight of the classification loss in the total loss function, default 0.5.
# Weight of the classification loss in the total loss function, affecting the importance of
# correct class prediction relative to other components.
mixup: float = 0.3  # Blends two images and their labels, creating a composite image
device = "mps"
## TRAINING
# %%
project = f"pcb_{model_name}_3fold_epochs_{epochs}_batch_{batch}"  # save results to project/name

start_time = time.time()
fold_times = []
results = {}
for k in range(ksplit):
    dataset_yaml = ds_yamls[k]
    model.train(
        data=dataset_yaml,
        epochs=epochs,
        batch=batch,
        lr0=0.001,
        lrf=0.0001,
        imgsz=imgsz,
        save_period=save_period,
        verbose=verbose,
        project=project,
        mixup=mixup,
    )
    results[k] = model.metrics  # save output metrics for further analysis
end_time = time.time()

end_time = time.time()
total_time_spent = end_time - start_time

print(
    f"Training completed in {total_time_spent:.2f} seconds, which is equivalent to {total_time_spent / 60:.2f} minutes"
)
for i, fold_time in enumerate(fold_times, start=1):
    print(
        f"Time spent on fold {i}: {fold_time:.2f} seconds, which is equivalent to {fold_time / 60:.2f} minutes"
    )
