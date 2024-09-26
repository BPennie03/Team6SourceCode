import os
import re
import utils
from ultralytics import YOLO


def main():
    model_path = utils.get_most_recent_version(
        'train', 'runs/detect/')
    model = YOLO(f'{model_path}/weights/best.pt')

    print(f'Using {model_path}/weights/best.pt')

    for file in os.listdir('test_images'):
        model.predict(f'test_images/{file}', save=True, imgsz=640, conf=0.7)


if __name__ == "__main__":
    main()
