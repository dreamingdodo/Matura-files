import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import wandb

def evaluate_model(model, val_loader, processor):
    model.eval()
    all_predictions, all_labels = [], []

    with torch.no_grad():
        for inputs, answers in val_loader:
            input_ids = inputs["input_ids"]
            pixel_values = inputs["pixel_values"]

            outputs = model.generate(input_ids=input_ids, pixel_values=pixel_values, max_new_tokens=1024, num_beams=3)
            generated_text = processor.batch_decode(outputs, skip_special_tokens=False)

            predictions = [processor.post_process_generation(gt, task="<OD>") for gt in generated_text]
            all_predictions.extend(predictions)
            all_labels.extend(answers)

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')

    wandb.log({"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1})

    return accuracy, precision, recall, f1
