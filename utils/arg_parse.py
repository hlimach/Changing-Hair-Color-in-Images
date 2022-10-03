import argparse

def parse_args(params):
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, required=True, help='number of epochs')
    parser.add_argument('--dataroot', type=str, default='./dataset', help='path to root folder of dataset content')
    parser.add_argument('--data_folder', type=str, default='CelebA-HQ-img', help='name of data folder containing images')
    parser.add_argument('--setA_file', type=str, default='hair_list_A.json', help='name of json file for set A')
    parser.add_argument('--setB_file', type=str, default='hair_list_B.json', help='name of json file for set B')
    parser.add_argument('--dataset_type', type=str, default='train', help='type of dataset (train, test)')
    parser.add_argument('--split_point', type=int, default=12000, help='index of splitting train and test sets')

    parser.add_argument('--K', type=int, required=True, help='number of image samples to draw from each domain')

    args = parser.parse_args(params)
    return args