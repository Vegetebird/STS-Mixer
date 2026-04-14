import os
import torch
import logging
import torchvision
import numpy as np
from torch import nn
from tqdm import tqdm
import torch.utils.data
from common.utils import *
import torch.nn.functional as F
from model.stsmixer import Model
from common.msr import MSRAction3D
from common.arguments import parse_args


def train(model, criterion, optimizer, lr_scheduler, data_loader):
    model.train()

    for i, data in enumerate(tqdm(data_loader, dynamic_ncols=True)):
        x, target, _  = data
        x, target = x.cuda(), target.cuda()

        output = model(x)

        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        lr_scheduler.step()

def test(model, criterion, data_loader):
    model.eval()
    video_prob = {}
    video_label = {}

    for i, data in enumerate(tqdm(data_loader, dynamic_ncols=True)):
        clip, target, video_idx = data

        clip = clip.cuda()
        target = target.cuda()

        output = model(clip)

        prob = F.softmax(input=output, dim=1)

        batch_size = clip.shape[0]
        target = target.cpu().numpy()
        video_idx = video_idx.cpu().numpy()
        prob = prob.cpu().numpy()

        for i in range(0, batch_size):
            idx = video_idx[i]
            if idx in video_prob:
                video_prob[idx] += prob[i]
            else:
                video_prob[idx] = prob[i]
                video_label[idx] = target[i]

    video_pred = {k: np.argmax(v) for k, v in video_prob.items()}
    pred_correct = [video_pred[k]==video_label[k] for k in video_pred]
    total_acc = np.mean(pred_correct)

    return total_acc * 100


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dataset = MSRAction3D(root=args.data_path, frames_per_clip=args.frames, step_between_clips=1, num_points=args.num_points, train=True)
    dataset_test = MSRAction3D(root=args.data_path, frames_per_clip=args.frames, step_between_clips=1, num_points=args.num_points, train=False)

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)

    model = Model(args)
    model.cuda()

    lr = args.lr
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)

    warmup_iters = args.lr_warmup_epochs * len(data_loader)
    lr_milestones = [len(data_loader) * m for m in args.lr_milestones]
    lr_scheduler = WarmupMultiStepLR(optimizer, milestones=lr_milestones, gamma=args.lr_gamma, warmup_iters=warmup_iters, warmup_factor=1e-5)

    if args.test:
        Load_model(args.checkpoint, model)

        acc_all = []
        with torch.no_grad():
            acc = test(model, criterion, data_loader_test)

        print(f'Acc: {acc:.2f}')
        exit()

    acc, best_acc, best_epoch, save_name = 0, 0, 0, ''
    for epoch in range(1, args.epochs+1):
        train(model, criterion, optimizer, lr_scheduler, data_loader)

        with torch.no_grad():
            acc = test(model, criterion, data_loader_test)

        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
            save_name = save_checkpoint(args.checkpoint, model, best_epoch, best_acc, save_name)

        lr = optimizer.param_groups[0]['lr']
        logging.info(f'Epoch {epoch}, lr {lr:.6f}, acc {acc:.2f}, {best_epoch}: {best_acc:.2f}')
        print(f'Epoch {epoch}, lr {lr:.6f}, acc {acc:.2f}, {best_epoch}: {best_acc:.2f}')

if __name__ == "__main__":
    args = parse_args()

    main(args)
