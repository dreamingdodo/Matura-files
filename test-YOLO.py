import cv2
import os
import supervision as sv
from ultralytics import YOLOv10

# Load the model
model = YOLOv10(f'runs/detect/train14/weights/best.pt')

# Directory containing images
image_folder = 'img'

# Output directory for annotated images
output_folder = 'output'


# Initialize annotators
bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Process each image in the folder
for filename in os.listdir(image_folder):
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(image_folder, filename)
        
        # Read the image
        image = cv2.imread(image_path)
        
        # Get model predictions
        results = model(image)[0]
        detections = sv.Detections.from_ultralytics(results)
        
        # Annotate the image
        annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)
        
        # Save the annotated image
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, annotated_image)

print("Processing complete.")

