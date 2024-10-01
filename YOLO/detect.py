import argparse
import os
import utils
from ultralytics import YOLO


def get_model():
    model_path = utils.get_most_recent_version('train', 'runs/detect/')
    return YOLO(f'{model_path}/weights/best.pt')


def detect(dir_path='test_images'):
    model = get_model()
    results = model.predict(dir_path, show_boxes=False, imgsz=640, conf=0.7)
    for r in results:
        if r.boxes:
            r.save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO Detection Script")
    parser.add_argument('-f', type=str, default='test_images',
                        help='Path to the directory containing images for detection')
    args = parser.parse_args()

    test_input = args.f
    detect(test_input)
