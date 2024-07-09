import torch
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm import tqdm
import os
import wandb

def train_model(train_loader, val_loader, model, processor, epochs=10, lr=1e-6):
    optimizer = AdamW(model.parameters(), lr=lr)
    num_training_steps = epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    # Initialize wandb
    wandb.init(project="groceries-object-detection")
    wandb.watch(model, log="all")

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for inputs, answers in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}"):

            input_ids = inputs["input_ids"]
            pixel_values = inputs["pixel_values"]
            labels = processor.tokenizer(
                text=answers,
                return_tensors="pt",
                padding=True,
                return_token_type_ids=False
            ).input_ids.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

            outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        print(f"Average Training Loss: {avg_train_loss}")

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, answers in tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}/{epochs}"):

                input_ids = inputs["input_ids"]
                pixel_values = inputs["pixel_values"]
                labels = processor.tokenizer(
                    text=answers,
                    return_tensors="pt",
                    padding=True,
                    return_token_type_ids=False
                ).input_ids.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

                outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
                loss = outputs.loss

                val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            print(f"Average Validation Loss: {avg_val_loss}")

        # Log losses to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
        })

        # Save model checkpoint
        output_dir = f"./model_checkpoints/epoch_{epoch+1}"
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        processor.save_pretrained(output_dir)

    # Finish wandb run
    wandb.finish()

if __name__ == "__main__":
    import argparse
    from model import load_model
    from dataset import get_dataloaders
    from download_data import download_dataset

    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument('--train_file_path', type=str, required=True, help='Path to the training dataset files')
    parser.add_argument('--val_file_path', type=str, required=True, help='Path to the validation dataset files')
    args = parser.parse_args()

    dataset = download_dataset(args.train_file_path, args.val_file_path)
    train_loader, val_loader = get_dataloaders(dataset, train_file_path=args.train_file_path, val_file_path=args.val_file_path)
    model, processor = load_model()

    EPOCHS = 10
    LR = 5e-6
    train_model(train_loader, val_loader, model, processor, epochs=EPOCHS, lr=LR)
