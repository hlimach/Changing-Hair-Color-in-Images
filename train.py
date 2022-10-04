from data import *
from utils import *

if __name__ == '__main__':
    params = parse_args(['--epochs', '12',
                         '--K', '12',
                         '--L', '100',
                         ])
    data = HairDataset(params)