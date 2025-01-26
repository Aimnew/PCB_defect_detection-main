import cv2
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from pcb.utils import get_subfolder


def visualize_annotations(
    image_name: str, images_dir: str, annot_df: pd.DataFrame, is_subfolder: bool = False
) -> plt.Figure:
    """
    Visualize annotations on the image
    :param image_name: filename of the image
    :param images_dir: directory containing the images
    :param annot_df: dataframe containing information about the annotations
    :param is_subfolder: whether the image is in a subfolder
    :return: figure of annotated image
    """

    # Construct path for image
    if is_subfolder:
        image_path = Path(images_dir) / get_subfolder(image_name) / image_name
    else:
        image_path = Path(images_dir) / image_name

    # Read image
    image = cv2.imread(str(image_path))

    # Filter annotations for the current image
    annotations = annot_df[annot_df["filename"] == image_name]

    # Draw bounding boxes on the image
    for _, annot in annotations.iterrows():
        xmin, ymin, xmax, ymax = (
            annot["xmin"],
            annot["ymin"],
            annot["xmax"],
            annot["ymax"],
        )
        class_label = annot["class"]

        # Check if confidence column exists
        confidence = annot.get("confidence")
        if confidence is not None:
            class_label += f" ({confidence:.2f})"

        color = (255, 255, 255)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 3)

        # Add background to the text
        text_size = cv2.getTextSize(class_label, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 2)[0]
        cv2.rectangle(
            image,
            (xmin, ymin - text_size[1] - 5),
            (xmin + text_size[0], ymin - 1),
            color,
            -1,
        )

        # Add text
        cv2.putText(
            image,
            class_label,
            (xmin, ymin - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 0, 0),
            2,
        )

    # Convert BGR image to RGB (Matplotlib expects RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Plot the image with annotations
    plt.figure(figsize=(18, 10))
    plt.imshow(image_rgb)
    plt.axis("off")
    plt.title("Annotations")
    plt.text(
        10,
        image_rgb.shape[0] + 100,
        f"Image: {image_name}",
        color="black",
        fontsize=11,
        ha="left",
    )
    plt.show()

    return image
