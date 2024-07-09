import torch
from transformers import AutoModelForCausalLM, AutoProcessor

CHECKPOINT = "microsoft/Florence-2-base-ft"
REVISION = 'refs/pr/6'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    model = AutoModelForCausalLM.from_pretrained(CHECKPOINT, trust_remote_code=True, revision=REVISION).to(DEVICE)
    processor = AutoProcessor.from_pretrained(CHECKPOINT, trust_remote_code=True, revision=REVISION)
    return model, processor

def run_inference(model, processor, image_path, task="<OD>", text="<OD>"):
    from PIL import Image
    image = Image.open(image_path)

    inputs = processor(text=text, images=image, return_tensors="pt").to(DEVICE)
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    response = processor.post_process_generation(generated_text, task=task, image_size=(image.width, image.height))

    return response

if __name__ == "__main__":
    model, processor = load_model()
    response = run_inference(model, processor, "dog.jpeg")
    print(response)
