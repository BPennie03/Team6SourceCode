import argparse
import shutil
from ultralytics import YOLO
import os
import utils
import red_circle_detect

DETECT_DIR = 'output/detect_results/'


def get_model():
    model_path = utils.get_most_recent_version('train', 'runs/detect')
    print(f"Using model from {model_path}")
    return YOLO(f'{model_path}/weights/best.pt')


def detect(dir_path='test_images'):
    model = get_model()
    results = model.predict(dir_path, show_boxes=False, imgsz=640, conf=0.7)
    for r in results:
        if r.boxes:  # if the result has a bounding box
            # r.save(conf=False, boxes=False)
            r.save()
            move_results_files()


def move_results_files(src_dir='.', dest_dir=DETECT_DIR):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    for file_name in os.listdir(src_dir):
        if file_name.startswith('results_'):
            shutil.move(os.path.join(src_dir, file_name),
                        os.path.join(dest_dir, file_name))


def clear_output_dir(dir_path=DETECT_DIR):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO Detection Script")
    parser.add_argument('-f', type=str, default='resources',
                        help='Path to the directory containing images for detection')
    args = parser.parse_args()

    test_input = args.f
    clear_output_dir()
    detect(test_input)
    # red_circle_detect.process_images(DETECT_DIR) will fix this later
