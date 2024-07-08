import os
import json
import shutil
import xml.etree.ElementTree as ET

def move_and_remove_directories(destination_directory):
    # Define source directories
    directories = [
        "cake", "candy", "cereal", "chips", "chocolate", "coffee", "corn", "fish",
        "flour", "honey", "jam", "juice", "milk", "nuts", "oil", "pasta", "rice",
        "soda", "spices", "sugar", "tea", "tomato_sauce", "vinegar", "water", "beans"
    ]

    # Move files from source directories to the destination directory
    for source_dir in directories:
        folder_name = os.path.join(destination_directory, source_dir)
        for filename in os.listdir(folder_name):
            source_path = os.path.join(folder_name, filename)
            destination_path = os.path.join(destination_directory, filename)
            shutil.move(source_path, destination_path)
        #remove source dir
        shutil.rmtree(folder_name)

    print(f"Files moved to {destination_directory}, and source directories removed.")

def normalize_coordinates(x, y, image_width, image_height):
    # Normalize coordinates to [0, 1]
    norm_x = x / image_width
    norm_y = y / image_height
    return norm_x, norm_y

def pascalvoc_to_florence2(pascalvoc_folder, images_folder, output_jsonl):
    with open(output_jsonl, "w") as jsonl_file:
        for filename in os.listdir(pascalvoc_folder):
            if filename.endswith(".xml"):
                # Read PascalVOC XML file
                xml_path = os.path.join(pascalvoc_folder, filename)
                tree = ET.parse(xml_path)
                root = tree.getroot()

                # Get image name (remove ".xml" extension)
                image_name = os.path.splitext(filename)[0]
                image_path = os.path.join(images_folder, f"{image_name}.jpg")

                # Initialize Florence-2 format dictionary
                florence2_annotations = {
                    "image": image_name,
                    "prefix": "<OD>",
                    "suffix": ""
                }

                # Iterate over object annotations
                for obj in root.findall("object"):
                    class_name = obj.find("name").text
                    bbox = obj.find("bndbox")
                    x1 = int(bbox.find("xmin").text)
                    y1 = int(bbox.find("ymin").text)
                    x2 = int(bbox.find("xmax").text)
                    y2 = int(bbox.find("ymax").text)

                    # Normalize and scale coordinates
                    image_width, image_height = 224, 224
                    norm_x1, norm_y1 = normalize_coordinates(x1, y1, image_width, image_height)
                    norm_x2, norm_y2 = normalize_coordinates(x2, y2, image_width, image_height)
                    loc_x1, loc_y1 = int(norm_x1 * 1000), int(norm_y1 * 1000)
                    loc_x2, loc_y2 = int(norm_x2 * 1000), int(norm_y2 * 1000)

                    # Construct Florence-2 suffix
                    suffix = f"{class_name}<loc_{loc_x1}><loc_{loc_y1}><loc_{loc_x2}><loc_{loc_y2}>"
                    florence2_annotations["suffix"] += suffix

                # Write to JSONL file
                jsonl_file.write(json.dumps(florence2_annotations) + "\n")

# do the stuff
move_and_remove_directories("/groceries-object-detection-dataset/dataset/train/images/")
move_and_remove_directories('/groceries-object-detection-dataset/dataset/train/annotations/')
pascalvoc_folder = "/groceries-object-detection-dataset/dataset/train/annotations/"
images_folder = "/groceries-object-detection-dataset/dataset/train/images/"
output_jsonl = "/groceries-object-detection-dataset/dataset/train/images/annotations_train.jsonl"
pascalvoc_to_florence2(pascalvoc_folder, images_folder, output_jsonl)
move_and_remove_directories('/groceries-object-detection-dataset/dataset/val/images/')
move_and_remove_directories('/groceries-object-detection-dataset/dataset/val/annotations/')
pascalvoc_folder = "/groceries-object-detection-dataset/dataset/val/annotations/"
images_folder = "/groceries-object-detection-dataset/dataset/val/images/"
output_jsonl = "/groceries-object-detection-dataset/dataset/val/images/annotations_val.jsonl"
pascalvoc_to_florence2(pascalvoc_folder, images_folder, output_jsonl)

print(f"PascalVOC annotations converted to Florence-2 format and saved as {output_jsonl}")
