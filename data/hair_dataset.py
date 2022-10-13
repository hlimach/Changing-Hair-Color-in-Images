import os
import json
import random
from torch.utils.data import Dataset
from utils.data_processing import *

class HairDataset(Dataset):
    def __init__(self, params):
        super().__init__()
        self.params = params

        # load set A and B json files
        set_A_file = open(os.path.join(params.dataroot, self.params.setA_file))
        set_B_file = open(os.path.join(params.dataroot, self.params.setB_file))

        self.set_A = json.load(set_A_file)
        self.set_B = json.load(set_B_file)

        set_A_file.close()
        set_B_file.close()

        # set train and test file sizes according to type of set
        if self.params.dataset_type == 'train':
            self.set_A = self.set_A[:self.params.split_point]
            self.set_B = self.set_B[:self.params.split_point]
        elif self.params.dataset_type == 'test':
            self.set_A = self.set_A[self.params.split_point:]
            self.set_B = self.set_B[self.params.split_point:]
        else:
            raise Exception('params.dataset_type is neither train nor test')

        self.set_A_size = len(self.set_A)
        self.set_B_size = len(self.set_B)

        assert(self.set_A_size == self.set_B_size)

        # define transforms function 
        self.transforms = get_transforms(self.params)
        
    
    def __len__(self):
        return len(self.set_A_size)

    def __getitem__(self, index):
        # randomly sample K images from both set A and set B
        k_input_images = random.sample(self.set_A, self.params.K) # set of possible original images to be input into generator
        k_target_colors = random.sample(self.set_B, self.params.K) # set of possible target hair colors to be input into generator
        
        i = 0 
        random.shuffle(k_input_images)
        best_color_dist = get_color_distances(k_input_images, k_target_colors)

        # maximize distance between input image and target hair color
        for l in range(self.params.L):
            random.shuffle(k_target_colors)
            current_color_dist = get_color_distances(k_input_images, k_target_colors)
            if current_color_dist > best_color_dist:
                i = random.randint(0, self.params.K - 1)
                best_color_dist = current_color_dist
                best_target_color = k_target_colors[i]

        A_entry = k_input_images[i]
        B_entry = self.set_B[random.randint(0, self.set_B_size - 1)]

        # open image A and B
        images_path = os.path.join(self.params.dataroot, self.params.data_folder)
        img_A_path = os.path.join(images_path, A_entry[0])
        img_B_path = os.path.join(images_path, B_entry[0])
        img_A = Image.open(img_A_path).convert('RGB')
        img_B = Image.open(img_B_path).convert('RGB')

        # generate hair color images for image A, B, and target
        A_rgb = create_image_from_rgb(A_entry[1:-1])
        B_rgb = create_image_from_rgb(B_entry[1:-1])
        target_rgb = create_image_from_rgb(best_target_color[1:-1])

        # apply transformations to all images and save
        out = {'A': self.transforms(img_A),
                'B': self.transforms(img_B),
                'A_rgb': self.transforms(A_rgb),
                'B_rgb': self.transforms(B_rgb),
                'target_rgb': self.transforms(target_rgb)}
        return out
        