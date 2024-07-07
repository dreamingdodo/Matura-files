import os
import glob
import xml.etree.ElementTree as ET

def convert_annotation(xml_file, classes):
    # Parse the XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Get the size of the image
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)

    # Create a new TXT file
    txt_file = os.path.splitext(xml_file)[0] + ".txt"
    with open(txt_file, 'w') as out_file:
        # Iterate over each object in the XML file
        for obj in root.iter('object'):
            cls = obj.find('name').text
            if cls not in classes:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (
                (float(xmlbox.find('xmin').text) + float(xmlbox.find('xmax').text)) / (2 * width),
                (float(xmlbox.find('ymin').text) + float(xmlbox.find('ymax').text)) / (2 * height),
                (float(xmlbox.find('xmax').text) - float(xmlbox.find('xmin').text)) / width,
                (float(xmlbox.find('ymax').text) - float(xmlbox.find('ymin').text)) / height
            )
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in b]) + '\n')

    # Remove the original XML file
    os.remove(xml_file)

classes = ["__background__", "beans", "cake", "candy", "cereal", "chips", "chocolate", "coffee", "corn", "fish", "flour", "honey", "jam", "juice", "milk", "nuts", "oil", "pasta", "rice", "soda", "spices", "sugar", "tea", "tomato_sauce", "vinegar", "water"]

# Specify the directory containing your XML files
xml_dir = "groceries-object-detection-dataset/dataset/train/images/"

# Convert each XML file in the directory
for xml_file in glob.glob(os.path.join(xml_dir, "*.xml")):
    convert_annotation(xml_file, classes)
