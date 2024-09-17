import cv2
import os
from matplotlib import pyplot as plt

def get_next_filename(image_name):
    i = 0
    while os.path.exists(image_name + "%s.png" % i):
        i += 1
    
    return image_name + "%s.png" % i



def detect_and_outline(input_path, classifier_path, output_path):
    """Detects and outlines stop signs in an image.

    Args:
        image_path (string): Path to the image file.
        classifier_path (string): Path to the Haar cascade classifier.
        output_path (string): Path where the output image will be saved.
    """

    """ use .walk() or .crawl() to iterate through all images in usb drive"""
    for root, dirs, files in os.walk(input_path):
        for filename in files:
            if filename.endswith(".png"):
                print(f'Processing image: {os.path.join(root, filename)}')
    
                img = cv2.imread(os.path.join(root, filename))
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                classifier = cv2.CascadeClassifier(classifier_path)

                found = classifier.detectMultiScale(img_gray, minSize=(20, 20))
                # """put found images in new folder, not crop shutilmove"""
                if len(found) != 0:
                    plt.imshow(img_rgb)
                    plt.axis('off')

                    output_image_path = get_next_filename(output_path +"output_image")
                    plt.savefig(output_image_path)
                    print(f'Output image saved to: {output_image_path}')


if __name__ == "__main__":
    img_path = os.getcwd()[:len(os.getcwd())-3] + "resources/images"
    classifier_path = os.getcwd()[:len(os.getcwd())-3] + "resources/stop_signs.xml"
    output_path = os.getcwd() + "/output/"  # Adjusted to match the volume mount
    detect_and_outline(img_path, classifier_path, output_path)
