import argparse
import cv2
import numpy as np
import os
import utils


DETECT_DIR = 'output/detect_results/'


def detect_and_crop(image_path):
    """Function to detect red circles in an image and crop them

    Args:
        image_path (str): path to the image to process
    """
    print(f"Processing image: {image_path}...")
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Could not read image {image_path}")
        return

    # Convert original image to BGR, then blur to reduce noise
    captured_frame_bgr = cv2.cvtColor(original_image, cv2.COLOR_BGRA2BGR)
    captured_frame_bgr = cv2.medianBlur(captured_frame_bgr, 3)

    # Convert to Lab color space, we only need to check one channel (a-channel) for red here
    captured_frame_lab = cv2.cvtColor(captured_frame_bgr, cv2.COLOR_BGR2Lab)

    # Define lower and upper bounds for the red color (in HSV)
    lower_red = np.array([10, 150, 150])
    upper_red = np.array([190, 255, 255])

    # Threshold the HSV image to get only red colors
    mask1 = cv2.inRange(captured_frame_lab, lower_red, upper_red)
    mask1 = cv2.GaussianBlur(mask1, (5, 5), 2, 2)

    mask = mask1

    # Find contours in the mask
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check if any contours are found
    if contours:

        num_circles = len(contours)

        for idx, contour in enumerate(contours):
            # Get the bounding box of the contour
            x, y, w, h = cv2.boundingRect(contour)

            # Crop the original image using the bounding box
            cropped_image = original_image[y:y+h, x:x+w]

            #! Skip small images, not needed if our "circles" are dots...
            # if cropped_image.shape[:2] < (25, 25):
            #     continue

            # Resize the cropped image to a fixed size
            cropped_image = cv2.resize(cropped_image, (200, 200))

            # Save the cropped image with a new name
            filename = os.path.basename(image_path)
            name, ext = os.path.splitext(filename)
            cropped_filename = f"{name}_cropped_{idx}{ext}"

            # Create folder if it doesn't exist
            folder = os.path.join('output/red_circle_results/')
            os.makedirs(folder, exist_ok=True)

            # Save cropped image in the "cropped images" folder
            cropped_image_path = os.path.join(folder, cropped_filename)
            cv2.imwrite(cropped_image_path, cropped_image)
    else:
        print(f"No red circle found in {image_path}\n")

    print(f"{num_circles} red circle(s) found in {image_path}\n")


def process_images(dir_path):
    """Function to process all images in a directory by passing them into the detect_and_crop function

    Args:
        dir_path (str): path to the directory containing images
    """
    files = os.listdir(dir_path)

    image_files = [file for file in files if file.endswith(
        ('.png', '.jpg', '.jpeg', '.webp'))]

    for image_file in image_files:
        image_path = os.path.join(dir_path, image_file)
        detect_and_crop(image_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Red Circle Detection Script")
    parser.add_argument('-f', type=str, default=DETECT_DIR,
                        help='Path to the directory containing images for detection')
    args = parser.parse_args()

    test_input = args.f
    utils.clear_output_dir(DETECT_DIR)

    print('Starting red circle detection...')
    print('=====================================')
    process_images(test_input)
