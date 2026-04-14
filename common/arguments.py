import argparse
import os
import math
import time
import torch
import socket
import sys
import logging
import argparse

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='STSMixer')
    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('--dataset', default='msr', type=str)
    parser.add_argument('--data-path', default='data/MSRAction3D/Depth/processed_data', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--checkpoint', default='data/checkpoint')
    parser.add_argument('--frames', default=24, type=int, metavar='N')
    parser.add_argument('--num-points', default=2048, type=int, metavar='N')
    parser.add_argument('--batch_size', default=24, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--workers', default=10, type=int, metavar='N')
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
    parser.add_argument('--weight-decay', default=1e-4, type=float, metavar='W')
    parser.add_argument('--lr-milestones', nargs='+', default=[20, 30], type=int)
    parser.add_argument('--lr-gamma', default=0.1, type=float)
    parser.add_argument('--lr-warmup-epochs', default=10, type=int)

    args = parser.parse_args()

    if not args.test:
        logtime = time.strftime('%m%d_%H%M_%S')
        args.checkpoint = args.checkpoint + '/' + logtime

        red   = "\033[1;31m%s\033[0m"
        green = "\033[1;32m%s\033[0m"
        blue  = "\033[1;34m%s\033[0m"

        print(green % args.checkpoint)

        os.makedirs(args.checkpoint, exist_ok=True)

        logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', \
            filename=os.path.join(args.checkpoint, 'train.log'), level=logging.INFO)

        args_write = dict((name, getattr(args, name)) for name in dir(args)
            if not name.startswith('_'))

        file_name = os.path.join(args.checkpoint, 'configs.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('==> Args:\n')
            for k, v in sorted(args_write.items()):
                opt_file.write('  %s: %s\n' % (str(k), str(v)))
            opt_file.write('==> Args:\n')

    return args


