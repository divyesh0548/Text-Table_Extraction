from ultralyticsplus import YOLO, render_result
from PIL import Image
import torch.serialization
import ultralytics

with torch.serialization.safe_globals([ultralytics.nn.tasks.DetectionModel]):
    model = YOLO('foduucom/table-detection-and-extraction')

# set model parameters
model.overrides['conf'] = 0.25  # NMS confidence threshold
model.overrides['iou'] = 0.45  # NMS IoU threshold
model.overrides['agnostic_nms'] = False  # NMS class-agnostic
model.overrides['max_det'] = 1000  # maximum number of detections per image

# set image
image_path = 'page_1.png'
image = Image.open(image_path).convert("RGB")

# perform inference
results = model.predict(image)

print("Detected Boxes:", results[0].boxes)

# observe results
render = render_result(model=model, image=image_path, result=results[0])  # Render result from model
render.show()
