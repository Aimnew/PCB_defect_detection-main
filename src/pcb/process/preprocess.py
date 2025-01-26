from pathlib import Path
import cv2
import pandas as pd
import numpy as np


def resize_images(
    input_dir: str | Path, 
    output_dir: str | Path, 
    target_size: tuple = (640, 640)
) -> None:
    """
    Resize images with improved preprocessing
    """
    for image_path in input_dir.rglob("*.jpg"):
        image = cv2.imread(str(image_path))
        
        # Улучшить качество изображения
        image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)  # Убрать шум
        
        # Улучшить контраст
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
        # Resize с сохранением пропорций
        h, w = image.shape[:2]
        ratio = min(target_size[0]/w, target_size[1]/h)
        new_size = (int(w*ratio), int(h*ratio))
        resized = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        
        # Создать холст нужного размера
        canvas = np.full((target_size[1], target_size[0], 3), (114,114,114), dtype=np.uint8)
        
        # Разместить изображение по центру
        x_offset = (target_size[0] - new_size[0]) // 2
        y_offset = (target_size[1] - new_size[1]) // 2
        canvas[y_offset:y_offset+new_size[1], x_offset:x_offset+new_size[0]] = resized
        
        cv2.imwrite(str(Path(output_dir) / image_path.name), canvas)


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
