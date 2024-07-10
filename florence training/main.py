import torch
import os
from setup import setup_env
setup_env()
from download import download_example_image, download_and_extract_roboflow, download_dataset
from model import load_model, run_inference
from train import train_model
from evaluate import evaluate_model
from dataset import get_dataloaders
from transformers import AutoProcessor, AutoModelForCausalLM
import wandb
import argparse

def main():
    parser = argparse.ArgumentParser(description="Train or evaluate the model")
    parser.add_argument('--batch_size', type=int, default=15, help='Batch size for training and validation')
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate'], required=True, help='Mode: train or evaluate')
    parser.add_argument('--checkpoint_path', type=str, help='Path to the model checkpoint for evaluation')
    args = parser.parse_args()

    if args.mode == 'train':
        # Download example data and dataset
        download_example_image()
        download_dataset()
        dataset = 'groceries-object-detection-dataset'

        train_loader, val_loader, _ = get_dataloaders(dataset, batch_size=args.batch_size, mode=args.mode)
        model, processor = load_model()

        EPOCHS = 10
        LR = 5e-6
        train_model(train_loader, val_loader, model, processor, epochs=EPOCHS, lr=LR)

    elif args.mode == 'evaluate':
        if not args.checkpoint_path:
            raise ValueError("Checkpoint path must be provided for evaluation mode")

        # Load the model and processor from the checkpoint
        CHECKPOINT = args.checkpoint_path
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = AutoModelForCausalLM.from_pretrained(CHECKPOINT, trust_remote_code=True).to(DEVICE)
        processor = AutoProcessor.from_pretrained(CHECKPOINT, trust_remote_code=True)

        # Initialize wandb
        wandb.init(project="groceries-object-detection-eval")

        # Download and evaluate test dataset only if required
        _, _, test_loader = get_dataloaders(None, batch_size=args.batch_size, mode=args.mode)
        if test_loader:
            accuracy, precision, recall, f1 = evaluate_model(model, test_loader, processor)
            print(f"Model Evaluation - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
        else:
            print("No test dataset found. Skipping evaluation.")

        wandb.finish()

if __name__ == "__main__":
    main()
