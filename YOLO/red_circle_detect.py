import cv2
import numpy as np
import os

# Function to detect red circles and crop the images


def detect_and_crop(image_path):
    print(f"Processing image: {image_path}")

    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Could not read image {image_path}")
        return

    # Convert original image to BGR, since Lab is only available from BGR
    captured_frame_bgr = cv2.cvtColor(original_image, cv2.COLOR_BGRA2BGR)

    # First blur to reduce noise prior to color space conversion
    captured_frame_bgr = cv2.medianBlur(captured_frame_bgr, 3)

    # Convert to Lab color space, we only need to check one channel (a-channel) for red here
    captured_frame_lab = cv2.cvtColor(captured_frame_bgr, cv2.COLOR_BGR2Lab)

    # Define lower and upper bounds for the red color (in HSV)
    lower_red = np.array([10, 150, 150])
    upper_red = np.array([190, 255, 255])

    # Threshold the HSV image to get only red colors
    mask1 = cv2.inRange(captured_frame_lab, lower_red, upper_red)
    mask1 = cv2.GaussianBlur(mask1, (5, 5), 2, 2)

    # Combine the masks
    mask = mask1

    # Find contours in the mask
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check if any contours are found
    if contours:
        for idx, contour in enumerate(contours):
            # Get the bounding box of the contour
            x, y, w, h = cv2.boundingRect(contour)

            # Crop the original image using the bounding box
            cropped_image = original_image[y:y+h, x:x+w]

            # Skip small images
            if cropped_image.shape[:2] < (50, 50):
                continue

            # Resize the cropped image to a fixed size
            cropped_image = cv2.resize(cropped_image, (100, 100))

            # Save the cropped image with a new name
            filename = os.path.basename(image_path)
            name, ext = os.path.splitext(filename)
            cropped_filename = f"{name}_cropped_{idx}{ext}"

            # Create folder if it doesn't exist
            folder = os.path.join(os.path.dirname(__file__), '../output/red_circle_results/')
            os.makedirs(folder, exist_ok=True)

            # Save cropped image in the "cropped images" folder
            cropped_image_path = os.path.join(folder, cropped_filename)
            cv2.imwrite(cropped_image_path, cropped_image)
    else:
        print(f"No red circle found in {image_path}")


def process_images(dir_path):
    if os.path.exists(dir_path):
        files = os.listdir(dir_path)

    # Filter only image files
    image_files = [file for file in files if file.endswith(
        ('.png', '.jpg', '.jpeg', '.webp'))]

    # sort the files in ascending order using natural sorting (idk how tf this works)
    # image_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))

    # Process each image
    for image_file in image_files:
        image_path = os.path.join(dir_path, image_file)
        detect_and_crop(image_path)
