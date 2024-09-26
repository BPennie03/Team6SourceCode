import argparse
import os
import utils
from ultralytics import YOLO


def get_model():
    model_path = utils.get_most_recent_version('train', 'runs/detect/')
    return YOLO(f'{model_path}/weights/best.pt')


def detect(dir_path='test_images'):
    model = get_model()
    for file in os.listdir(dir_path):
        model.predict(f'{dir_path}/{file}', save=True, imgsz=640, conf=0.7)


def single_detect(file_path):
    model = get_model()
    model.predict(file_path, save=True, imgsz=640, conf=0.7)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO Detection Script")
    parser.add_argument('-f', type=str, default='test_images',
                        help='Path to the directory containing images for detection')
    args = parser.parse_args()

    test_input = args.f
    if os.path.isdir(test_input):
        detect(test_input)
    else:
        single_detect(test_input)
