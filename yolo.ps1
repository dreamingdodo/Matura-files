Add-Type -AssemblyName System.Windows.Forms

# Function to open the folder browser dialog
function Select-FolderDialog {
    $folderBrowser = New-Object System.Windows.Forms.FolderBrowserDialog
    $folderBrowser.ShowDialog() | Out-Null
    return $folderBrowser.SelectedPath
}

# Function to open the file open dialog
function Select-FileDialog {
    $fileBrowser = New-Object System.Windows.Forms.OpenFileDialog
    $fileBrowser.Filter = "Model files (*.pt)|*.pt"
    $fileBrowser.ShowDialog() | Out-Null
    return $fileBrowser.FileName
}

# Output directory for results
$resultsDir = "$PWD/runs/detect/predict"

# Check if the results directory exists, if not, create it
if (-Not (Test-Path -Path $resultsDir)) {
    New-Item -ItemType Directory -Path $resultsDir
}

# Select the model path
$modelPath = Select-FileDialog
if (-not $modelPath) {
    Write-Output "Model path selection cancelled."
    exit
}

# Select the source directory
$sourceDir = Select-FolderDialog
if (-not $sourceDir) {
    Write-Output "Source directory selection cancelled."
    exit
}

# Install necessary tools and packages
winget install --id Git.Git -e --source winget
pip install -q git+https://github.com/THU-MIG/yolov10.git
pip install -q supervision roboflow

# Download images from GitHub repository recursively
git clone --depth 1 https://github.com/aleksandar-aleksandrov/groceries-object-detection-dataset $sourceDir

# Run YOLO detection on each image in the source directory recursively
$imageFormats = @(".jpg", ".png", ".jpeg")
Get-ChildItem -Path $sourceDir -Recurse -Filter "*.png" | ForEach-Object {
    $imagePath = $_.FullName
    yolo task=detect mode=predict conf=0.25 save=True model=$modelPath source=$imagePath
}

Write-Output "Results saved to $resultsDir" 
