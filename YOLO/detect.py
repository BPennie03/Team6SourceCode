import os
from ultralytics import YOLO

model = YOLO('runs/detect/train/weights/best.pt')
for file in os.listdir('test_images'):
    model.predict(f'test_images/{file}', save=True, imgsz=640, conf=0.7)
