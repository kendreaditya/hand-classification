# %%
import cv2
import os
from tqdm import tqdm

# %%
def frame_extractor(file_name, save_folder):
    vidcap = cv2.VideoCapture(file_name)
    success_read, image = vidcap.read()
    count = 0
    while tqdm(success_read):
        success_write = cv2.imwrite(os.path.join(save_folder, f'frame-{count}.jpg'), image)     # save frame as JPEG file      
        success_read, image = vidcap.read()
        count += 1

# %%
frame_extractor('./fist.mp4', './fist/')

# %%
frame_extractor('./open_hand.mp4', './open_hand/')