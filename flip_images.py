import os
from PIL import Image
import re

# These flips were determined by hand, looking through the images in Real-IAD
flip_dict = {
    'audiojack': [0, 1, 0, 0, 0],
    'bottle_cap': [0, 0, 0, 0, 0],
    'button_battery': [0, 1, 1, 1, 1],
    'end_cap': [0, 1, 1, 0, 0],
    'eraser': [1, 0, 0, 0, 0],
    'fire_hood': [0, 1, 1, 0, 0],
    'mint': [0, 1, 1, 1, 0],
    'mounts': [1, 0, 0, 0, 0],
    'pcb': [0, 1, 0, 0, 0],
    'phone_battery': [0, 1, 1, 1, 0],
    'plastic_nut': [0, 1, 1, 0, 0],
    'plastic_plug': [0, 1, 1, 1, 0],
    'porcelain_doll': [1, 0, 0, 0, 1],
    'regulator': [0, 1, 1, 1, 1],
    'rolled_strip_base': [1, 0, 0, 0, 0],
    'sim_card_set': [0, 1, 1, 0, 0],
    'switch': [0, 1, 1, 1, 1],
    'tape': [1, 0, 0, 0, 0],
    'terminalblock': [1, 0, 0, 0, 0],
    'toothbrush': [0, 1, 1, 1, 0],
    'toy': [0, 1, 1, 1, 1],
    'toy_brick': [1, 0, 0, 0, 1],
    'transistor1': [0, 1, 1, 1, 1],
    'u_block': [1, 0, 0, 1, 1],
    'usb': [0, 1, 0, 0, 0],
    'usb_adaptor': [0, 1, 1, 0, 0],
    'vcpill': [1, 0, 0, 0, 1],
    'wooden_beads': [0, 1, 0, 0, 0],
    'woodstick': [0, 1, 1, 0, 0],
    'zipper': [0, 1, 1, 0, 0]
}

# Specify the directory with subdirectories
directory = '/data/anomaly_detection/realiad/classes'

def get_camera_number(filename):
    match = re.search(r'C(\d)', filename)
    if match:
        return int(match.group(1))
    else:
        raise ValueError("A weird file has been found?")

def should_flip(subdirectory, filename): 
    return flip_dict[subdirectory][get_camera_number(filename) - 1] == 1

def flip_images(subdirectory):
    subdirectory_path = os.path.join(directory, subdirectory)
    
    if os.path.isdir(subdirectory_path):
        for root, dirs, files in os.walk(subdirectory_path):
            for file in files:
                if file.endswith('.jpg') or file.endswith(".png"):
                    filename = os.path.join(root, file)
                    if should_flip(subdirectory, file):
                        print("Flipping: ", filename)
                        with open(filename, 'rb') as f_in:
                            image = Image.open(f_in)
                            flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)
                        with open(filename, 'wb') as f_out:
                            flipped_image.save(f_out)


answer = input("Are you sure you want to flip images in Real-IAD? This will change mane files IN-PLACE on your system! (y/n)")

if answer == "y":
    print("Starting procedure...")
else:
    print("Please write 'y' as answer to proceed. Exiting...")
    exit()
    
for subdirectory in os.listdir(directory):
    flip_images(subdirectory)
