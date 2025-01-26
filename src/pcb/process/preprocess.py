from pathlib import Path
import cv2
import pandas as pd


def resize_images(
    input_dir: str | Path, 
    output_dir: str | Path, 
    target_size: tuple = (640, 640)
) -> None:
    """
    Resize images in the input directory and save them to the output directory
    :param input_dir: input directory containing images
    :param output_dir: output directory for resized images
    :param target_size: target size for resized images
    :return: None
    """
    # Create the output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Convert input_dir to Path if it's not already
    input_dir = Path(input_dir)

    for image_path in input_dir.rglob("*.jpg"):
        # Read the image - convert Path to string for cv2.imread
        image = cv2.imread(str(image_path))

        # Resize the image
        resized_image = cv2.resize(image, target_size)

        # Save the resized image to the output folder
        output_path = Path(output_dir) / image_path.name
        cv2.imwrite(str(output_path), resized_image)


def resize_annotations(
    annot_df: pd.DataFrame, target_size: tuple = (640, 640)
) -> pd.DataFrame:
    """
    Resize the bounding box coordinates in the annotation DataFrame
    :param annot_df: DataFrame containing annotations
    :param target_size: target size of new image
    :return: DataFrame with resized annotations
    """
    all_data = []

    # Iterate through the annotation DataFrame
    for index, row in annot_df.iterrows():
        # Resize the bounding box coordinates
        width_ratio = target_size[0] / row["width"]
        height_ratio = target_size[1] / row["height"]

        resized_xmin = int(row["xmin"] * width_ratio)
        resized_ymin = int(row["ymin"] * height_ratio)
        resized_xmax = int(row["xmax"] * width_ratio)
        resized_ymax = int(row["ymax"] * height_ratio)

        # Update the all process list with resized annotations
        all_data.append(
            {
                "filename": row["filename"],
                "width": target_size[0],
                "height": target_size[1],
                "class": row["class"],
                "xmin": resized_xmin,
                "ymin": resized_ymin,
                "xmax": resized_xmax,
                "ymax": resized_ymax,
            }
        )

    annot_df_resized = pd.DataFrame(all_data)
    return annot_df_resized
