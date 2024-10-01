import argparse
import os
import utils
from ultralytics import YOLO
import shutil


def get_model():
    model_path = utils.get_most_recent_version('train', 'runs/detect/')
    return YOLO(f'{model_path}/weights/best.pt')


def detect(dir_path='test_images'):
    model = get_model()
    results = model.predict(dir_path, show_boxes=False, imgsz=640, conf=0.7)
    for r in results:
        if r.boxes:
            r.save(conf=False, boxes=False)
            move_results_files()


def move_results_files(src_dir='.', dest_dir='detect_results'):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    for file_name in os.listdir(src_dir):
        if file_name.startswith('results_'):
            shutil.move(os.path.join(src_dir, file_name),
                        os.path.join(dest_dir, file_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO Detection Script")
    parser.add_argument('-f', type=str, default='test_images',
                        help='Path to the directory containing images for detection')
    args = parser.parse_args()

    test_input = args.f
    detect(test_input)
