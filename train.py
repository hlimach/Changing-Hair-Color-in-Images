from data import *
from utils import *

if __name__ == '__main__':
    params = ['--epochs', '12',
                '--K', '2']
    args = parse_args(params)
    data = HairDataset(args)