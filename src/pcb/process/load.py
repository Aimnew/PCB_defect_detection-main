import xml.etree.ElementTree as ET  # ToDo: rewrite using defusedxml package - package not safe


def parse_xml(xml_file: str) -> list[list[str : str | int | float]]:
    """
    Parse xml file and return a list of dictionaries containing the process
    :param xml_file:
    :return:
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    data = []

    filename = root.find("filename").text
    width = int(root.find("size/width").text)
    height = int(root.find("size/height").text)

    for obj in root.findall("object"):
        name = obj.find("name").text
        xmin = int(obj.find("bndbox/xmin").text)
        ymin = int(obj.find("bndbox/ymin").text)
        xmax = int(obj.find("bndbox/xmax").text)
        ymax = int(obj.find("bndbox/ymax").text)

        data.append(
            {
                "filename": filename,
                "width": width,
                "height": height,
                "class": name,
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
            }
        )

    return data
