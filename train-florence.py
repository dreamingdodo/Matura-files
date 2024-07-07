import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_scheduler, AutoModelForCausalLM, AutoProcessor
from PIL import Image
import wandb
from tqdm.auto import tqdm
import requests

# Set device to GPU if available, otherwise use CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define your dataset class
class DetectionDataset(Dataset):
    def __init__(self, jsonl_file_path: str, image_directory_path: str):
        self.entries = []
        self.jsonl_file_path = jsonl_file_path
        self.image_directory_path = image_directory_path
        self._load_entries()

    def _load_entries(self):
        with open(self.jsonl_file_path, 'r') as file:
            for line in file:
                data = json.loads(line)
                self.entries.append(data)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        image_path = os.path.join(self.image_directory_path, entry['image'] + '.png')
        try:
            image = Image.open(image_path)
            return image, entry
        except FileNotFoundError:
            raise FileNotFoundError(f"Image file {image_path} not found.")

# Initialize wandb for experiment tracking
wandb.init(project='object_detection_project', entity='your_username')

# Define your training function
def train_model(train_loader, val_loader, model, epochs=10):
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for inputs, answers in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}"):
            # Your training loop here

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, answers in tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}/{epochs}"):
                # Your validation loop here

        # Log metrics to wandb
        wandb.log({"train_loss": train_loss / len(train_loader), "val_loss": val_loss / len(val_loader)})

        # Save model checkpoints (optional)
        output_dir = f"./model_checkpoints/epoch_{epoch+1}"
        os.makedirs(output_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(output_dir, 'model.pth'))

# Define your datasets and DataLoader instances
dataset_dir = "/content/groceries-object-detection-dataset/dataset"
train_dataset = DetectionDataset(
    jsonl_file_path = os.path.join(dataset_dir, 'train/images/annotations_train.jsonl'),
    image_directory_path = os.path.join(dataset_dir, 'train/images/')
)
val_dataset = DetectionDataset(
    jsonl_file_path = os.path.join(dataset_dir, 'val/images/annotations_val.jsonl'),
    image_directory_path = os.path.join(dataset_dir, 'val/images/')
)

BATCH_SIZE = 16
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Load pretrained Florence-2-large model and processor
model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True).to(DEVICE)
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)

# Run training
EPOCHS = 50
train_model(train_loader, val_loader, model, epochs=EPOCHS)

# Finish wandb run
wandb.finish()
