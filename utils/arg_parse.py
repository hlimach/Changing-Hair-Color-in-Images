import argparse

def parse_args(params):
    parser = argparse.ArgumentParser()

    # dataset loading and setup parameters
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
    parser.add_argument('--flip', default=False, action='store_true', help='random horizontal flip of image [True | False]')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size for datatset')

    # model parameters
    parser.add_argument('--nf', type=int, default=64, help='number of filters')
    parser.add_argument('--r_blocks', type=int, default=9, help='number of residual blocks in generator')
    parser.add_argument('--n_layers', type=int, default=3, help='number of layers in basic discriminator')

    # training parameters
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for ADAM optimizer')
    parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy [linear | step]')
    parser.add_argument('--beta1', type=float, default=0.5, help='momentum of ADAM optimizer')
    parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs with the initial learning rate')
    parser.add_argument('--n_epochs_decay', type=int, default=100, help='number of epochs to linearly decay learning rate to zero')
    parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations for stepLR')
    parser.add_argument('--lambda_cyc', type=float, default=5.0, help='weight for cycle loss')
    parser.add_argument('--lambda_idt', type=float, default=0.333, help='weight for identity loss')
    
    # model training, loading and saving options
    parser.add_argument('--continue_train', default=False, action='store_true', help='if training should be continued from a checkpoint')
    parser.add_argument('--print_iter_interval', type=int, default=50, help='print training stats after n iterations in each epoch')
    parser.add_argument('--save_dir', type=str, default='./model/checkpoints', help='where model checkpoints are saved')
    parser.add_argument('--img_pool_size', type=int, default=50, help='how many images to save while training')


    args = parser.parse_args(params)
    return args