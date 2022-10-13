import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# taken from original repo
def create_image_from_rgb(color, height = 256, width = 256):
    colors_array = np.array(color)
    colors_array = np.reshape(colors_array,(1,1,3))
    img_array = np.ones((height,width,3)) * colors_array
    img = Image.fromarray(np.uint8(img_array * 255))
    return img

def get_color_distances(input, target):
    # recheck
    dist = 0.0
    for x, y in zip(input, target):
        dist +=  np.linalg.norm(np.subtract(x[1:-1],y[1:-1]),ord=2)
    return dist

def get_transforms(params):
    transform_list = []

    if 'resize' in params.preprocess:
        transform_list.append(transforms.Resize([params.scale_size, params.scale_size], 
                                transforms.InterpolationMode.BICUBIC))

    if 'crop' in params.preprocess:
        transform_list.append(transforms.RandomCrop(params.crop_size))
        
    if params.flip:
        transform_list.append(transforms.RandomHorizontalFlip())

    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

    return transforms.Compose(transform_list)