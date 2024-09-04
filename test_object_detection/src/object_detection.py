import cv2
import os
from matplotlib import pyplot as plt


def detect_and_outline(image_path, classifier_path, output_path):
    """Detects and outlines stop signs in an image.

    Args:
        image_path (string): Path to the image file.
        classifier_path (string): Path to the Haar cascade classifier.
        output_path (string): Path where the output image will be saved.
    """

    """ use .walk() or .crawl() to iterate through all images in usb drive"""
    """use an if statement to check for .png file type"""
    print(f'Processing image: {image_path}')
    
    img = cv2.imread(image_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    classifier = cv2.CascadeClassifier(classifier_path)

    found = classifier.detectMultiScale(img_gray, minSize=(20, 20))
    """put found images in new folder, not crop"""
    if len(found) != 0:
        for (x, y, width, height) in found:
            cv2.rectangle(img_rgb, (x, y), (x + height,
                          y + width), (0, 255, 0), 5)

    plt.imshow(img_rgb)
    plt.axis('off')
    plt.savefig(output_path)
    print(f'Output image saved to: {output_path}')


if __name__ == "__main__":
    img_path = "../resources/images/sign3.png"
    classifier_path = "../resources/stop_signs.xml"
    output_path = "output/output_image.png"
    detect_and_outline(img_path, classifier_path, output_path)
