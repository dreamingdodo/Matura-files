#!/bin/bash

# Set up environment variables
export ROBOFLOW_API_KEY='api_key'
export HUGGINGFACE_MODEL_URL='https://huggingface.co/dreamingdodo/YOLOv10-groceries/resolve/main/best.pt'

# Install required libraries
pip install git+https://github.com/THU-MIG/yolov10.git
pip install supervision roboflow opencv-python

# Download the model weights from Hugging Face
wget -O best.pt $HUGGINGFACE_MODEL_URL

# Create a Python script to run the model
cat <<EOF > run_yolov10.py
import os
import cv2
import supervision as sv
from subprocess import call
import os
from urllib.request import urlretrieve
from ultralytics import YOLOv10
from roboflow import Roboflow

# Ensure the API key is set
api_key = os.getenv('ROBOFLOW_API_KEY')
if not api_key:
    raise ValueError("ROBOFLOW_API_KEY environment variable not set")

# Initialize Roboflow
rf = Roboflow(api_key=api_key)
project = rf.workspace("matura").project("groceries-detection-ojjwk")
version = project.version(1)
dataset = version.download("yolov9")

# Download dataset
if os.path.exists(freiburg_groceries_dataset):
    print("Dataset found")
else:

  dataset_url = "http://aisdatasets.informatik.uni-freiburg.de/freiburg_groceries_dataset/freiburg_groceries_dataset.tar.gz"
  
  print("Downloading dataset.")
  urlretrieve(dataset_url, "../freiburg_groceries_dataset.tar.gz")
  print("Extracting dataset.")
  call(["tar", "-xf", "../freiburg_groceries_dataset.tar.gz", "-C", "../"])
  os.remove("../freiburg_groceries_dataset.tar.gz")
  print("Done.")

# Load the model
model = YOLOv10('best.pt')

# Load the dataset (annotations and yaml dont matter here)
dataset = sv.DetectionDataset.from_yolo(
    images_directory_path="freiburg_groceries_dataset/dataset/val/images/",
    annotations_directory_path="freiburg_groceries_dataset/dataset/val/images",
    data_yaml_path="freiburg_groceries_dataset/dataset/classes.txt" 
)

# Annotators
BoxAnnotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Process all images
for path, image, annotation in dataset:
    results = model(source=image, conf=0.25)[0]
    detections = sv.Detections.from_ultralytics(results)
    annotated_image = BoxAnnotator.annotate(scene=image, detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)
    output_path = os.path.join("output", os.path.basename(path))
    cv2.imwrite(output_path, annotated_image)
    print(f"Processed and saved: {output_path}")

EOF

# Create output directory if it doesn't exist
mkdir -p output

# Run the Python script
python run_yolov10.py

