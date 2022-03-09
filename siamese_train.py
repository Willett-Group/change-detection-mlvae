"""
Train siamese network with
[method in Ruslan 2015 paper or contrastive method] and [cnn or resnet]
"""
import argparse
import glob
import json
import logging
import random
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image

import datasets
import networks
import utils

parser = argparse.ArgumentParser()
parser.add_argument("--datapath", type=str, default="../data", help="location of the data corpus")
parser.add_argument("--dataset", type=str, default="mnist", help="which dataset")
parser.add_argument("--n_max", type=int, default=1000, help="numbe of time series samples")
parser.add_argument("--t_max", type=int, default=50, help="number of timestamps in a time series sample")
parser.add_argument("--p_max", type=int, default=50, help="number of pairs to sample within a time series sample")
parser.add_argument("--method", type=str, default="ruslan", help="method on siamese outputs")
parser.add_argument("--arch", type=str, default="cnn", help="siamese architecture")
parser.add_argument("--lr", type=float, default=0.001, help="initial adam learning rate")

parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--report_freq", type=float, default=50, help="report frequency")
parser.add_argument("--seed", type=int, default=0, help="random seed for reproducible test dataset and results")
parser.add_argument("--num_workers", type=int, default=4, help="number of workers")
parser.add_argument("--output_dir", type=str, default="DEBUG", help="output directory")
parser.add_argument("--resume", type=bool, default=False)
parser.add_argument("--resume_dir", type=str)

args = parser.parse_args()
utils.create_exp_dir(args.output_dir, scripts_to_save=glob.glob("*.py"))


# logging
log_format = "%(asctime)s %(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt="%m/%d %I:%M:%S %p")
fh = logging.FileHandler(Path(args.output_dir, "log.txt"))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
# tensorboard writer
writer = SummaryWriter(args.output_dir)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def train(train_queue, model, criterion, optimizer, epoch):
    model.train()
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()

    for step, (input, target) in enumerate(train_queue):
        n = input.size(0)
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        target = target.float()

        optimizer.zero_grad()
        output = model(input)
        output = torch.flatten(output)
        p = torch.sigmoid(output)
        logits = torch.stack((1-p, p), dim=1)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        prec1, = utils.accuracy(logits, target, topk=(1,))
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)

        if step % args.report_freq == 0:
            logging.info("train %03d loss %e top1 %f", step, objs.avg, top1.avg)
            writer.add_scalar("LossBatch/train", objs.avg, epoch * len(train_queue) + step)
            writer.add_scalar("AccuBatch/train", top1.avg, epoch * len(train_queue) + step)

        writer.add_scalar("LossEpoch/train", objs.avg, epoch)
        writer.add_scalar("AccuEpoch/train", top1.avg, epoch)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion, epoch):
    model.eval()
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()

    for step, (input, target) in enumerate(valid_queue):
        n = input.size(0)
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        target = target.float()

        output = model(input)
        output = torch.flatten(output)
        p = torch.sigmoid(output)
        logits = torch.stack((1 - p, p), dim=1)
        loss = criterion(output, target)

        prec1, = utils.accuracy(logits, target, topk=(1,))
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)

        if step % args.report_freq == 0:
            logging.info("valid %03d loss %e top1 %f", step, objs.avg, top1.avg)
            writer.add_scalar("LossBatch/valid", objs.avg, epoch * len(valid_queue) + step)
            writer.add_scalar("AccuBatch/valid", top1.avg, epoch * len(valid_queue) + step)

        writer.add_scalar("LossEpoch/valid", objs.avg, epoch)
        writer.add_scalar("AccuEpoch/valid", top1.avg, epoch)

    return top1.avg, objs.avg


def main():
    logging.info("args = %s", args)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)
    cudnn.benchmark = True
    cudnn.enabled = True

    classes = {
        "mnist": range(10),
        "cifar10": range(10),
        "cifar100": range(100),
        "celeba": [4, 9, 17, 20, 24]
        # "celeba" : [3]
    }
    train_data_con = datasets.CON(
        datapath=args.datapath,
        dataset=args.dataset,
        split="train",
        n_max=args.n_max,
        t_max=args.t_max,
        p_max=args.p_max,
        classes=classes[args.dataset],
        transform=utils.transforms[args.dataset])
    
    # visualize some examples
    image_list = [train_data_con[n][0]
    for n in range(0 * train_data_con.P, 1 * train_data_con.P)]
    images = torch.cat(image_list, dim=0)
    save_image(make_grid(images, nrow=train_data_con.P), Path(args.output_dir, f"CON.png"))

    image_list = [utils.drawlines(train_data_con.get_x_n(n), train_data_con.splits[n][0][0])
    for n in range(0, 5)]
    images = torch.cat(image_list, dim=0)
    save_image(make_grid(images, nrow=train_data_con.T), Path(args.output_dir, f"TS.png"))


    valid_data_con = datasets.CON(
        datapath=args.datapath,
        dataset=args.dataset,
        split="train",
        n_max=int(0.2*args.n_max),
        t_max=args.t_max,
        p_max=args.p_max,
        classes=classes[args.dataset],
        transform=utils.transforms[args.dataset])
    train_queue_con = DataLoader(
        train_data_con,
        shuffle=True,
        num_workers=args.num_workers,
        batch_size=args.batch_size)
    valid_queue_con = DataLoader(
        valid_data_con,
        shuffle=False,
        num_workers=args.num_workers,
        batch_size=args.batch_size)
    
    if args.dataset in ["mnist", "cifar10", "cifar100"]:
        model = networks.SiameseNet32(loss=args.method, arch=args.arch).to(device)
    else:
        model = networks.SiameseNet128(loss=args.method, arch=args.arch).to(device)
    criterion = nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0
    is_best = False
    start_epoch = 0

    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()

        logging.info("epoch %d", epoch)
        train_acc, train_obj = train(train_queue_con, model, criterion, optimizer, epoch)
        logging.info("train_acc %f train_loss %e \n", train_acc, train_obj)
        with torch.no_grad():
            valid_acc, valid_obj = infer(valid_queue_con, model, criterion, epoch)
        if valid_acc > best_acc:
            best_acc = valid_acc
            is_best = True
        else:
            is_best = False
        logging.info("valid_acc %f best_acc %f valid_loss %e\n", valid_acc, best_acc, valid_obj)

        utils.save_checkpoint({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict()}, is_best, args.output_dir)

        end_time = time.time()
        duration = end_time - start_time
        logging.info("Epoch time %ds", duration)


if __name__ == "__main__":
    main()
