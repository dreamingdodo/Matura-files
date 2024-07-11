import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Tuple

class JSONLDataset:
    def __init__(self, jsonl_file_path: str, image_directory_path: str, mode: str):
        self.jsonl_file_path = jsonl_file_path
        self.image_directory_path = image_directory_path
        self.entries = self._load_entries()
        self.mode = mode

    def _load_entries(self) -> List[Dict[str, Any]]:
        entries = []
        with open(self.jsonl_file_path, 'r') as file:
            for line in file:
                data = json.loads(line)
                entries.append(data)
        return entries

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, Dict[str, Any]]:
        if idx < 0 or idx >= len(self.entries):
            raise IndexError("Index out of range")

        entry = self.entries[idx]
        image_filename = entry['image']
        if self.mode == 'train':
            image_filename += ".png"
        image_path = os.path.join(self.image_directory_path, image_filename)

        try:
            image = Image.open(image_path)
            return (image, entry)
        except FileNotFoundError:
            raise FileNotFoundError(f"Image file {image_path} not found.")

class DetectionDataset(Dataset):
    def __init__(self, jsonl_file_path: str, image_directory_path: str, mode: str):
        self.dataset = JSONLDataset(jsonl_file_path, image_directory_path, mode)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, data = self.dataset[idx]
        prefix = data['prefix']
        suffix = data['suffix']
        return prefix, suffix, image

def get_dataloaders(dataset, batch_size=6, num_workers=0, mode='train'):
    train_dataset = DetectionDataset(
        jsonl_file_path=f"groceries-object-detection-dataset/dataset/train/images/annotations_train.jsonl",
        image_directory_path=f"groceries-object-detection-dataset/dataset/train/images/",
        mode=mode
    )
    val_dataset = DetectionDataset(
        jsonl_file_path=f"groceries-object-detection-dataset/dataset/val/images/annotations_val.jsonl",
        image_directory_path=f"groceries-object-detection-dataset/dataset/val/images/",
        mode=mode
    )
    test_dataset = None
    if mode == 'evaluate':
        test_dataset = DetectionDataset(
            jsonl_file_path="test/test/annotations.jsonl",
            image_directory_path="test/test/",
            mode='test'
        )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_workers, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_workers) if test_dataset else None

    return train_loader, val_loader, test_loader

def collate_fn(batch):
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base-ft", trust_remote_code=True, revision='refs/pr/6')

    questions, answers, images = zip(*batch)
    inputs = processor(text=list(questions), images=list(images), return_tensors="pt", padding=True).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return inputs, answers

if __name__ == "__main__":
    from download_data import download_dataset
    dataset = download_dataset()
    train_loader, val_loader = get_dataloaders(dataset)
