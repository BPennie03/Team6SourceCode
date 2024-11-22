import argparse
import shutil
import os
import utils
from ultralytics import YOLO


DETECT_DIR = 'output/detect_results/'


def get_model():
    """Gets the most recent YOLO model

    Returns:
        YOLO: Most recent yolo model
    """
    model_path = utils.get_most_recent_version('train', 'runs/detect')
    print(f"Using model from {model_path}")
    return YOLO(f'{model_path}/weights/best.pt')


def detect(dir_path='resources'):
    """Detects objects in images in the specified directory. Only saves the 10
    images with the highest confidence

    Args:
        dir_path (str, optional): path to test images directory, defaults to 'resources'.
    """
    highest_conf = []

    model = get_model()
    results = model.predict(dir_path, show_boxes=False, imgsz=640, conf=0.5)

    for r in results:
        if r.boxes:
            max_conf = max([box.conf for box in r.boxes])
            highest_conf.append((max_conf, r))

    highest_conf = sorted(highest_conf, key=lambda x: x[0], reverse=True)[:10]
    for _, result in highest_conf:
        result.save()
        move_results_files()


def move_results_files(src_dir='.', dest_dir=DETECT_DIR):
    """Moves the results files to the output directory

    Args:
        src_dir (str, optional): source directory, defaults to '.'.
        dest_dir (str optional): directory to move files to, defaults to DETECT_DIR.
    """
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    for file_name in os.listdir(src_dir):
        if file_name.startswith('results_'):
            shutil.move(os.path.join(src_dir, file_name),
                        os.path.join(dest_dir, file_name))
    # Uncomment this to zip the results
    # shutil.make_archive('detect_results', 'zip', dest_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO Detection Script")
    parser.add_argument('-f', type=str, default='resources',
                        help='Path to the directory containing images for detection')
    args = parser.parse_args()

    test_input = args.f
    if not os.path.exists(test_input):
        print(f'Provided input path ({test_input}) does not exist')
        exit()  # terminate the script if the input path does not exist

    utils.clear_output_dir(DETECT_DIR)
    detect(test_input)
