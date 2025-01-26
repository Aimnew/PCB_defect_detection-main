import pandas as pd
import random
import shutil
import os
import pathlib


def convert_to_yolo_labels(annotation_df: pd.DataFrame, classes: list) -> list:
    """
    Convert annotations to YOLO format
    :param annotation_df: DataFrame containing annotations
    :param classes: list of class names
    :return: list of YOLO labels
    """
    yolo_labels = []

    for _, annot in annotation_df.iterrows():
        filename = annot["filename"]
        width, height = annot["width"], annot["height"]
        class_name = annot["class"]
        xmin, ymin, xmax, ymax = (
            annot["xmin"],
            annot["ymin"],
            annot["xmax"],
            annot["ymax"],
        )

        # Convert bounding box coordinates to YOLO format
        x_center = (xmin + xmax) / (2 * width)
        y_center = (ymin + ymax) / (2 * height)
        bbox_width = (xmax - xmin) / width
        bbox_height = (ymax - ymin) / height

        class_index = classes.index(class_name)

        # Append to YOLO labels list
        yolo_labels.append(
            (filename, class_index, x_center, y_center, bbox_width, bbox_height)
        )

    return yolo_labels


def read_yolo_labels_from_file(file_path: str | pathlib.Path) -> list:
    """
    Read YOLO labels from a file
    :param file_path:
    :return:
    """
    with open(file_path, "r") as file:
        labels = [[float(value) for value in line.strip().split()] for line in file]
    return labels


def split_images_and_labels(
    images_dir: pathlib.Path,
    labels: list,
    output_dir: pathlib.Path,
    train_split: float = 0.95,
    val_split: float = 0.05,
) -> None:
    """
    Split images and labels into train, validation, and test sets
    :param images_dir: input directory containing images
    :param labels: list of YOLO labels
    :param output_dir: output directory
    :param train_split: train split ratio
    :param val_split: validation split ratio
    :return:
    """
    # os.makedirs(output_dir, exist_ok=True)

    for folder in [
        "images/train",
        "images/val",
        "images/test",
        "labels/train",
        "labels/val",
        "labels/test",
    ]:
        (output_dir / folder).mkdir(parents=True, exist_ok=True)

    # Group labels by image filename
    image_labels = {}
    for label in labels:
        filename, class_index, x_center, y_center, bbox_width, bbox_height = label
        if filename not in image_labels:
            image_labels[filename] = []
        image_labels[filename].append(label)

    # Shuffle the image filenames
    image_filenames = list(image_labels.keys())
    random.shuffle(image_filenames)

    # Split the dataset
    num_images = len(image_filenames)
    num_train = int(num_images * train_split)
    num_val = int(num_images * val_split)

    train_filenames = image_filenames[:num_train]
    val_filenames = image_filenames[num_train : num_train + num_val]
    test_filenames = image_filenames[num_train + num_val :]

    # Write train, val, test images and labels
    for dataset, filenames in [
        ("train", train_filenames),
        ("val", val_filenames),
        ("test", test_filenames),
    ]:
        for filename in filenames:
            labels = image_labels[filename]
            label_file_path = (
                output_dir / f"labels/{dataset}/{os.path.splitext(filename)[0]}.txt"
            )
            with open(label_file_path, "a") as label_file:
                for label in labels:
                    _, class_index, x_center, y_center, bbox_width, bbox_height = label
                    label_file.write(
                        f"{class_index} {x_center} {y_center} {bbox_width} {bbox_height}\n"
                    )
            # Copy images to corresponding folders
            shutil.copy(
                src=images_dir / filename,
                dst=output_dir / f"images/{dataset}/{filename}",
            )


def yolo_to_original_annot(
    image_name: str, yolo_labels: list, annot_df: pd.DataFrame, classes: list[str]
) -> pd.DataFrame:
    """
    Convert YOLO labels to original annotations

    :param image_name: image filename
    :param yolo_labels: YOLO labels
    :param annot_df: Annotation DataFrame
    :param classes: list of class names
    :return: DataFrame containing original annotations
    """

    # Extract original width and height from annotation DataFrame
    original_size = annot_df.loc[
        annot_df["filename"] == image_name, ["width", "height"]
    ].iloc[0]
    original_width, original_height = original_size["width"], original_size["height"]

    # Convert yolo_labels to a DataFrame for vectorized operations
    yolo_df = pd.DataFrame(
        yolo_labels,
        columns=[
            "class_index",
            "x_center",
            "y_center",
            "bbox_width",
            "bbox_height",
            "confidence",
        ],
    )

    # Scale bounding box coordinates and dimensions to original size
    yolo_df["original_x_center"] = yolo_df["x_center"] * original_width
    yolo_df["original_y_center"] = yolo_df["y_center"] * original_height
    yolo_df["original_bbox_width"] = yolo_df["bbox_width"] * original_width
    yolo_df["original_bbox_height"] = yolo_df["bbox_height"] * original_height

    # Calculate original bounding box coordinates
    yolo_df["original_x_min"] = (
        yolo_df["original_x_center"] - yolo_df["original_bbox_width"] / 2
    )
    yolo_df["original_y_min"] = (
        yolo_df["original_y_center"] - yolo_df["original_bbox_height"] / 2
    )
    yolo_df["original_x_max"] = (
        yolo_df["original_x_center"] + yolo_df["original_bbox_width"] / 2
    )
    yolo_df["original_y_max"] = (
        yolo_df["original_y_center"] + yolo_df["original_bbox_height"] / 2
    )

    # Map class indices to class names
    yolo_df["class"] = yolo_df["class_index"].apply(lambda x: classes[int(x)])

    # Create the final DataFrame
    original_annot = yolo_df[
        [
            "class",
            "original_x_min",
            "original_y_min",
            "original_x_max",
            "original_y_max",
            "confidence",
        ]
    ].copy()
    original_annot["filename"] = image_name
    original_annot["width"] = int(original_width)
    original_annot["height"] = int(original_height)

    # Rename columns to match the expected output
    original_annot.rename(
        columns={
            "original_x_min": "xmin",
            "original_y_min": "ymin",
            "original_x_max": "xmax",
            "original_y_max": "ymax",
        },
        inplace=True,
    )

    # Convert bounding box coordinates to integers
    original_annot["xmin"] = original_annot["xmin"].astype(int)
    original_annot["ymin"] = original_annot["ymin"].astype(int)
    original_annot["xmax"] = original_annot["xmax"].astype(int)
    original_annot["ymax"] = original_annot["ymax"].astype(int)

    return original_annot
