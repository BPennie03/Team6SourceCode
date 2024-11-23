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

    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)

    # Define HSV range for red (covers both ends of the hue spectrum)
    lower_red1 = np.array([0, 50, 50])   # Lower range for red
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])  # Upper range for red
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    red_mask = cv2.GaussianBlur(red_mask, (5, 5), 2)

    contours, _ = cv2.findContours(
        red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    num_circles = 0
    for idx, contour in enumerate(contours):
        perimeter = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)
        if area == 0:
            continue
        circularity = (4 * np.pi * area) / (perimeter * perimeter)

        if 0.7 < circularity <= 1.2:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            if radius > 10:  # Filter out small circles
                mask_inside_circle = red_mask[int(
                    y - radius):int(y + radius), int(x - radius):int(x + radius)]
                inner_area = np.sum(mask_inside_circle) // 255

                if area > 2 * inner_area:
                    num_circles += 1

                    # Crop the circle region
                    x, y, w, h = cv2.boundingRect(contour)
                    cropped_image = original_image[y:y+h, x:x+w]

                    # Resize and save the cropped image
                    cropped_image = cv2.resize(cropped_image, (200, 200))
                    filename = os.path.basename(image_path)
                    name, ext = os.path.splitext(filename)
                    cropped_filename = f"{name}_cropped_{idx}{ext}"
                    folder = os.path.join('output/red_circle_results/')
                    os.makedirs(folder, exist_ok=True)
                    cropped_image_path = os.path.join(folder, cropped_filename)
                    cv2.imwrite(cropped_image_path, cropped_image)

    # Print result
    if num_circles > 0:
        print(f"{num_circles} hollow red circle(s) found in {image_path}\n")
    else:
        print(f"No hollow red circles found in {image_path}\n")


def process_images(dir_path=DETECT_DIR):
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
    utils.clear_output_dir('output/red_circle_results/')

    print('Starting red circle detection...')
    print('=====================================')
    process_images(test_input)
