# Install git for windows
winget install --id Git.Git -e --source winget
# Install necessary python packages
pip install -q git+https://github.com/THU-MIG/yolov10.git
pip install -q supervision roboflow

# Download the YOLO model from hugging face
$HUGGINGFACE_MODEL_URL = 'https://huggingface.co/dreamingdodo/YOLOv10-groceries/resolve/main/best.pt'
$localModelPath = "C:/Users/balme/Downloads/best.pt"
Invoke-WebRequest -Uri $HUGGINGFACE_MODEL_URL -OutFile $localModelPath

# Define the source directory for images
$sourceDir = "C:/Users/balme/Downloads/groceries-images"

# Download images from GitHub repository recursively
git clone --depth 1 https://github.com/aleksandar-aleksandrov/groceries-object-detection-dataset $sourceDir

# Run YOLO detection on each image in the source directory recursively
$imageFormats = "*.jpg,*.png,*.jpeg"
Get-ChildItem -Path $sourceDir -Recurse -Include $imageFormats | ForEach-Object {
    $imagePath = $_.FullName
    yolo task=detect mode=predict conf=0.25 save=True model=$localModelPath source=$imagePath
}

# Output directory for results
$resultsDir = "$PWD/runs/detect/predict"
Write-Output "Results saved to $resultsDir"
