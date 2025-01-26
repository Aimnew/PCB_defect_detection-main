import pandas as pd
from pathlib import Path


from pcb.process.load import parse_xml
from pcb.process.preprocess import resize_images, resize_annotations
from pcb.model.utils import convert_to_yolo_labels, split_images_and_labels

from pcb.utils import get_logger, get_dataset_dir

logger = get_logger(__name__)
# # %%
# root_dir = Path.cwd().parent.resolve()
# dataset_dir = root_dir / 'PCB_DATASET'
#
# # %%
# for dir_path in dataset_dir.rglob("*"):
#     if dir_path.is_dir():
#         logger.info(dir_path)
#
#
# # %%
# def count_files_in_folder(folder_path):
#     # Get list of all files in the folder
#     files = list(Path(folder_path).iterdir())
#
#     # Count the number of files
#     num_files = len(files)
#
#     return num_files
#
#
# # %%
# subfolders = ['Missing_hole', 'Mouse_bite', 'Open_circuit', 'Short', 'Spur', 'Spurious_copper']
#
# images_dir = dataset_dir / 'images'
# annot_dir = dataset_dir / 'Annotations'
#
# for subfolder in subfolders:
#     images_path = images_dir / subfolder
#     annot_path = annot_dir / subfolder
#
#     logger.info(f'{subfolder:<15} \t\
#             {count_files_in_folder(images_path)} images \t\
#             {count_files_in_folder(annot_path)} annotations')
#
# # %%
# # List to store parsed data from all XML files
# all_data = []
#
# # Recursively traverse subdirectories
# for xml_path in annot_dir.rglob("*.xml"):
#     all_data.extend(parse_xml(xml_path))
#
# annot_df = pd.DataFrame(all_data)
# annot_df.to_parquet(dataset_dir / 'annot_df.parquet')
#
# # %%
# annot_df.head()
#
# # %%
# image_name = '04_short_03.jpg'
# visualize_annotations(image_name, images_dir, annot_df, is_subfolder=True)
#
# # %%
# resized_img_dir = Path(dataset_dir, 'images_resized')
# resize_images(images_dir, resized_img_dir)
#
# # %%
# annot_df_resized = resize_annotations(annot_df)
# # %%
# annot_df_resized.head()
#
# ## Splitting the dataset
# # %%
# # Extract unique filenames
# output_dir = dataset_dir / 'output'
# output_dir.mkdir(parents=True, exist_ok=True)
#
# # %%
# classes = ['missing_hole', 'mouse_bite', 'open_circuit',
#            'short', 'spur', 'spurious_copper']
# yolo_labels = convert_to_yolo_labels(annot_df_resized, classes)
# split_images_and_labels(resized_img_dir, yolo_labels, output_dir)
#
# #%%
# yolo_labels[0]
#
#
# # K-Fold Cross Validation
# #%%
# dataset_path = Path(output_dir)
# labels = sorted(dataset_path.rglob("*labels/train/*.txt")) # all data in 'labels'
#
# #%%
# cls_idx = list(range(len(classes)))
# logger.info(list(zip(classes, cls_idx)))
#
#
# indx = [l.stem for l in labels] # uses base filename as ID (no extension)
# labels_df = pd.DataFrame([], columns=cls_idx, index=indx)
#
# for label in labels:
#     lbl_counter = Counter()
#
#     with open(label, 'r') as lf:
#         lines = lf.readlines()
#
#     for l in lines:
#         # classes for YOLO label uses integer at first position of each line
#         lbl_counter[int(l.split(' ')[0])] += 1
#
#     labels_df.loc[label.stem] = lbl_counter
#
# #%%
# labels_df = labels_df.fillna(0.0) # replace `nan` values with `0.0`
#  #%%
# labels_df.head()
#
# #%%
# ksplit = 3
# kf = KFold(n_splits=ksplit, shuffle=True, random_state=20)   # setting random_state for repeatable results
#
# kfolds = list(kf.split(labels_df))
#
# #%%
# folds = [f'split_{n}' for n in range(1, ksplit + 1)]
# folds_df = pd.DataFrame(index=indx, columns=folds)
#
# #%%
# for idx, (train, val) in enumerate(kfolds, start=1):
#     folds_df.loc[labels_df.iloc[train].index, f'split_{idx}'] = 'train'
#     folds_df.loc[labels_df.iloc[val].index, [f'split_{idx}']] = 'val'
#
# fold_lbl_distrb = pd.DataFrame(index=folds, columns=cls_idx)
#
# for n, (train_indices, val_indices) in enumerate(kfolds, start=1):
#     train_totals = labels_df.iloc[train_indices].sum()
#     val_totals = labels_df.iloc[val_indices].sum()
#
#     # To avoid division by zero, we add a small value (1E-7) to the denominator
#     ratio = val_totals / (train_totals + 1E-7)
#     fold_lbl_distrb.loc[f'split_{n}'] = ratio
# #%%
# fold_lbl_distrb
#
# #%%
# # Initialize a list to store image file paths
# images = sorted(dataset_path.rglob("*images/train/*.jpg"))
#
# # Create the necessary directories and dataset YAML files (unchanged)
# save_path = Path(dataset_path / f'{ksplit}fold_crossval')
# save_path.mkdir(parents=True, exist_ok=True)
# ds_yamls = []
#
# for split in folds_df.columns:
#     # Create directories
#     split_dir = save_path / split
#     split_dir.mkdir(parents=True, exist_ok=True)
#     (split_dir / 'train' / 'images').mkdir(parents=True, exist_ok=True)
#     (split_dir / 'train' / 'labels').mkdir(parents=True, exist_ok=True)
#     (split_dir / 'val' / 'images').mkdir(parents=True, exist_ok=True)
#     (split_dir / 'val' / 'labels').mkdir(parents=True, exist_ok=True)
#
#     # Create dataset YAML files
#     dataset_yaml = split_dir / f'{split}_dataset.yaml'
#     ds_yamls.append(dataset_yaml)
#
#     with open(dataset_yaml, 'w') as ds_y:
#         yaml.safe_dump({
#             'path': split_dir.as_posix(),
#             'train': 'train',
#             'val': 'val',
#             'names': classes
#         }, ds_y)
#
#
# #%%
# for image, label in zip(images, labels):
#     for split, k_split in folds_df.loc[image.stem].items():
#         # Destination directory
#         img_to_path = save_path / split / k_split / 'images'
#         lbl_to_path = save_path / split / k_split / 'labels'
#
#         # Copy image and label files to new directory
#         shutil.copy(image, img_to_path / image.name)
#         shutil.copy(label, lbl_to_path / label.name)
#
# #%%
# folds_df.to_csv(save_path / "kfold_datasplit.csv")
# fold_lbl_distrb.to_csv(save_path / "kfold_label_distribution.csv")


def preprocess(images_dir, annot_dir, dataset_dir):
    # List to store parsed data from all XML files
    all_data = []

    # Recursively traverse subdirectories
    for xml_path in annot_dir.rglob("*.xml"):
        all_data.extend(parse_xml(xml_path))

    annot_df = pd.DataFrame(all_data)
    annot_df.to_parquet(dataset_dir / "annotation.parquet")

    # Resize images and annotations
    resized_img_dir = Path(dataset_dir, "images_resized")
    # ToDo: add image size as parameter to be handed into preprocess
    resize_images(images_dir, resized_img_dir)

    annot_df_resized = resize_annotations(annot_df)

    output_dir = dataset_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # define classes
    classes = [
        "missing_hole",
        "mouse_bite",
        "open_circuit",
        "short",
        "spur",
        "spurious_copper",
    ]

    # create YOLO labels
    yolo_labels = convert_to_yolo_labels(annot_df_resized, classes)
    split_images_and_labels(resized_img_dir, yolo_labels, output_dir)

    logger.info("Done")


def main():
    # Get dataset directory
    dataset_dir = get_dataset_dir()
    images_dir = dataset_dir / "images" 
    annot_dir = dataset_dir / "Annotations"

    preprocess(dataset_dir=dataset_dir, images_dir=images_dir, annot_dir=annot_dir)
