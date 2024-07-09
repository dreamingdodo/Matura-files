import os
from setup import setup_env
setup_env()
from download import download_example_image, download_dataset
from model import load_model, run_inference
from train import train_model
from evaluate import evaluate_model
from dataset import get_dataloaders

def main():
    # Download example data and dataset
    download_example_image()
    dataset = download_dataset()

    # Load model and processor
    model, processor = load_model()

    # Prepare data loaders
    train_loader, val_loader = get_dataloaders(dataset)

    # Train model
    EPOCHS = 10
    LR = 5e-6
    train_model(train_loader, val_loader, model, processor, epochs=EPOCHS, lr=LR)

    # Evaluate model
    accuracy = evaluate_model(model, val_loader, processor)
    print(f"Model Accuracy: {accuracy}")

    # Run inference on example image
    response = run_inference(model, processor, "dog.jpeg")
    print(response)

if __name__ == "__main__":
    main()
