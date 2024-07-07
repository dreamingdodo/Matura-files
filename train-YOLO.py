# import wandb
# from wandb.integration.ultralytics import add_wandb_callback
import ultralytics
from ultralytics import YOLOv10

model = YOLOv10('yolov10m.pt')
# If you want to finetune the model with pretrained weights, you could load the 
# pretrained weights like below
# model = YOLOv10.from_pretrained('jameslahm/yolov10{n/s/m/b/l/x}')
# or
# wget https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10{n/s/m/b/l/x}.pt
# model = YOLOv10('yolov10{n/s/m/b/l/x}.pt')

# add_wandb_callback(model, enable_model_checkpointing=True)

model.train(data='matura.yaml', epochs=50, batch=16)

wandb.finish()

