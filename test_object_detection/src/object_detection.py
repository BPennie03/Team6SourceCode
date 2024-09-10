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
 
    # assign directory
    directory = input_path
 
    # iterate over files in
    # that directory
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith ((".png", ".jpg")):
                print(f'Processing image: {input_path+filename}')

                img = cv2.imread(os.path.join(root, filename))
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                classifier = cv2.CascadeClassifier(classifier_path)

                found = classifier.detectMultiScale(img_gray, minSize=(20, 20))

                if len(found) != 0:
                    for (x, y, width, height) in found:
                        cv2.rectangle(img_rgb, (x, y), (x + height,
                                        y + width), (0, 255, 0), 5)

                plt.imshow(img_rgb)
                plt.axis('off')


                plt.savefig(output_path + get_next_filename("output_image"))
                print(f'Output image saved to: {output_path}')


if __name__ == "__main__":
    img_path = "resources/images/"
    classifier_path = "resources/stop_signs.xml"
    output_path = "output/"  # Adjusted to match the volume mount
    detect_and_outline(img_path, classifier_path, output_path)
