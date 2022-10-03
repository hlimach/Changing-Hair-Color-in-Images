import os
import json
import random
from torch.utils.data import Dataset
from utils import *

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
        
        random.shuffle(k_input_images)
        random.shuffle(k_target_colors)

        current_input_image = k_input_images[0]
        current_target_color = k_target_colors[0]
        dist = get_distance(current_input_image, current_target_color)

        # image A, image B, hair color of image A, hair color of image B, target hair color sampled from domain B
        return super().__getitem__(index)
        