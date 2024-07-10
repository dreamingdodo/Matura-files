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
            image_size = pixel_values.shape[-2:]

            outputs = model.generate(input_ids=input_ids, pixel_values=pixel_values, max_new_tokens=1024, num_beams=3)
            print(f"Outputs: {outputs}")  # Print the raw outputs from the model

            generated_text = processor.batch_decode(outputs, skip_special_tokens=False)
            print(f"Generated text: {generated_text}")  # Print the decoded generated text

            predictions = [processor.post_process_generation(gt, task="<OD>", image_size=image_size) for gt in generated_text]
            print(f"Predictions: {predictions}")  # Print the post-processed predictions

            all_predictions.extend(predictions)
            all_labels.extend(answers)

    print(f"All predictions: {all_predictions}")  # Print all predictions after the loop
    print(f"All labels: {all_labels}")  # Print all labels after the loop

    # Convert all_predictions and all_labels to a format that sklearn metrics can handle
    all_predictions = [str(pred) for pred in all_predictions]
    all_labels = [str(label) for label in all_labels]

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=1)
    recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=1)
    f1 = f1_score(all_labels, all_predictions, average='weighted')

    print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")  # Print the metrics

    wandb.log({"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1})

    return accuracy, precision, recall, f1
