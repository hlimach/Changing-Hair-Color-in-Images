import argparse

def parse_args(params):
    parser = argparse.ArgumentParser()

    # dataset loading and setup params
    parser.add_argument('--epochs', type=int, required=True, help='number of epochs')
    parser.add_argument('--dataroot', type=str, default='./dataset', help='path to root folder of dataset content')
    parser.add_argument('--data_folder', type=str, default='CelebA-HQ-img', help='name of data folder containing images')
    parser.add_argument('--setA_file', type=str, default='hair_list_A.json', help='name of json file for set A')
    parser.add_argument('--setB_file', type=str, default='hair_list_B.json', help='name of json file for set B')
    parser.add_argument('--dataset_type', type=str, default='train', help='type of dataset (train, test)')
    parser.add_argument('--split_point', type=int, default=12000, help='index of splitting train and test sets')
    parser.add_argument('--K', type=int, required=True, help='number of image samples to draw from each domain')
    parser.add_argument('--L', type=int, required=True, help='number of times to shuffle target hair color set to maximize distance with input image')
    parser.add_argument('--preprocess', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop | crop | none]')
    parser.add_argument('--scale_size', type=int, default=286, help='scale images to this size')
    parser.add_argument('--crop_size', type=int, default=256, help='crop image to this size')
    parser.add_argument('--flip', type=bool, default=False, help='random horizontal flip of image [True | False]')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size for datatset')

    # model parameter
    parser.add_argument('--r_blocks', type=int, default=9, help='number of residual blocks in generator')
    parser.add_argument('--n_layers', type=int, default=3, help='number of layers in basic discriminator')

    args = parser.parse_args(params)
    return args