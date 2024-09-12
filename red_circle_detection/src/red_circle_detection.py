import cv2
import numpy as np
import os

# Function to detect red circles and crop the images


def detect_and_crop(image_path, cropped_images, filenames):
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

            # Resize the cropped image to a fixed size (e.g., 100x100)
            # Adjust the size as needed
            cropped_image = cv2.resize(cropped_image, (100, 100))

            cropped_images.append(cropped_image)
            filenames.append(os.path.basename(image_path))

            # Save the cropped image with a new name
            filename = os.path.basename(image_path)
            name, ext = os.path.splitext(filename)
            cropped_filename = f"{name}_cropped_{idx}{ext}"

            # Create folder if it doesn't exist
            folder = "../resources/cropped_images/"
            os.makedirs(folder, exist_ok=True)

            # Save cropped image in the "cropped images" folder
            cropped_image_path = os.path.join(folder, cropped_filename)
            cv2.imwrite(cropped_image_path, cropped_image)
    else:
        print(f"No red circle found in {image_path}")

# Process each image


def process_images():
    # List all files within Images folder
    folder = "../resources/images/"
    files = os.listdir(folder)

    # Filter only image files (you might need to adjust the extensions)
    image_files = [file for file in files if file.endswith(
        ('.png', '.jpg', '.jpeg'))]

    # sort the files in ascending order using natural sorting (idk how tf this works)
    image_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))

    cropped_images = []
    filenames = []

    # Process each image
    for image_file in image_files:
        image_path = os.path.join(folder, image_file)
        detect_and_crop(image_path, cropped_images, filenames)

    # Display cropped images in a table-like format
    display_images_in_table(cropped_images, filenames, cols=3)

# Function to display images in a table-like format


def display_images_in_table(images, filenames, cols=3):
    num_images = len(images)
    rows = int(np.ceil(num_images / cols))

    # Create a blank canvas to display the images
    canvas_height = 100 * rows
    canvas_width = 100 * cols
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    # Fill the canvas with images and labels
    current_row = 0
    current_col = 0
    for image, filename in zip(images, filenames):
        start_y = current_row * 100
        end_y = start_y + 100
        start_x = current_col * 100
        end_x = start_x + 100
        canvas[start_y:end_y, start_x:end_x] = image

        # Add filename label
        text_position = (start_x + 5, start_y + 20)
        cv2.putText(canvas, filename, text_position,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        current_col += 1
        if current_col == cols:
            current_col = 0
            current_row += 1

    # Display the canvas
    cv2.namedWindow('Cropped Images Table', cv2.WINDOW_NORMAL)
    cv2.imshow("Cropped Images Table", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def clear_folder(folder):
    print(f"Clearing folder: {folder}...")
    # List all files within the folder
    files = os.listdir(folder)

    # Remove each file
    for file in files:
        os.remove(os.path.join(folder, file))


if __name__ == "__main__":
    # Call the function to process the images
    clear_folder("../resources/cropped_images/")
    process_images()
