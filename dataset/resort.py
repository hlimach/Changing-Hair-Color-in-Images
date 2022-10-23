''' 
There are 30k images in the main folder itself 
this is creating issue of google drive timeout 
This script creates subfolders each containing 1k images
essentially 30 subfolders in the main folder.
'''

import os
import shutil

root = './CelebA-HQ-img'
images = os.listdir(root)
print(len(images))

for img in images:
    id = int(img.split('.')[0])
    folder = id - id % 1000
    folder_path = os.path.join(root, str(folder))
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    read_path = os.path.join(root, img)
    write_path = os.path.join(folder_path, img)
    shutil.move(read_path, write_path)
    print(('moved %s to subfolder %s') % (img, folder))
