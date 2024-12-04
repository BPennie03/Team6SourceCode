import argparse
import hashlib
import io
import os
import serial
import shutil
import time
import utils
import zipfile
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Util import Counter
from ultralytics import YOLO

DETECT_DIR = 'output/detect_results/'


def folder_to_byte_array(folder_path):
    """Converts a folder to a byte array by zipping it first."""

    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                zip_file.write(file_path, os.path.relpath(
                    file_path, folder_path))

    zip_buffer.seek(0)
    return zip_buffer.read()


def encrypt_message(key, plaintext):
    """Encrypts the given plaintext using AES CTR mode."""

    # Generate a random 16-byte initialization vector (IV)
    iv = os.urandom(16)

    # Create a counter object for CTR mode
    ctr = Counter.new(128, initial_value=int.from_bytes(iv, 'big'))

    # Create the AES cipher object
    cipher = AES.new(key, AES.MODE_CTR, counter=ctr)

    # Encrypt the plaintext
    ciphertext = cipher.encrypt(plaintext)

    # Return the IV and ciphertext as bytes
    return iv + ciphertext


def add_md5_checksum(byte_array):
    # Calculate the MD5 hash of the byte array
    md5_hash = hashlib.md5(byte_array).digest()

    # Append the MD5 hash to the original byte array
    return byte_array + md5_hash


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
        # result.save()
        result.save(conf=False, boxes=False)
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
    shutil.make_archive('detect_results', 'zip', dest_dir)


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

    print('Starting transmission...')
    aeskey = b'\x57\xc0\xb3\x51\x20\x72\x25\xa7\xeb\x44\x7b\x62\x61\x4f\xbd\x34\xeb\xd8\xab\x1a\xd4\x5d\x00\xae\x95\x10\x51\x75\x30\x4d\x19\x8c'

    print('Connecting...')
    ser = serial.Serial(port='/dev/ttyUSB0', baudrate=115200, parity=serial.PARITY_NONE,
                        stopbits=serial.STOPBITS_ONE, bytesize=serial.EIGHTBITS, timeout=1)
    img_path = os.getcwd()[:len(os.getcwd())-3] + "output/detect_results/"

    byte_array = folder_to_byte_array(img_path)

    with_md5 = add_md5_checksum(byte_array)

    encrypted = encrypt_message(aeskey, with_md5)
    print('Sending...')
    ser.write(byte_array)
    ser.write(b'\n')
    print('Transmission complete')
    ser.close()
