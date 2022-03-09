"""
Train MLVAE
"""
import argparse
import glob
import logging
import random
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image

import datasets
import networks
import utils

################################################################################
# things that make reconstructions worse (blurry, not able to see outlines...):
# too large or too small cs_dim
# too small number of training epochs
# larger beta values
parser = argparse.ArgumentParser()
parser.add_argument("--datapath", type=str, default="../data", help="location of the data corpus")
parser.add_argument("--dataset", type=str, default="cifar10", help="which dataset")
parser.add_argument("--n_max", type=int, default=1000, help="numbe of time series samples")
parser.add_argument("--t_max", type=int, default=50, help="number of timestamps in a time series sample")
parser.add_argument("--cs_dim", type=int, default=50, help="dimension of c, s")
parser.add_argument("--beta", type=float, default=1.0, help="beta value")
parser.add_argument("--w", type=float, default=1.0, help="weight for content")
parser.add_argument("--lr", type=float, default=0.001, help="adam initial learning rate")

parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=50)
parser.add_argument("--report_freq", type=float, default=200, help="report frequency")
parser.add_argument("--seed", type=int, default=0, help="random seed for reproducible test dataset and results")
parser.add_argument("--num_workers", type=int, default=4, help="number of workers")
parser.add_argument("--output_dir", type=str, default="DEBUG", help="output directory")
parser.add_argument("--resume", type=bool, default=False)
parser.add_argument("--resume_dir", type=str)

args = parser.parse_args()
assert args.batch_size == args.t_max
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

    train_data_ts = datasets.TS(
        datapath = args.datapath,
        dataset = args.dataset,
        split = "train",
        n_max = args.n_max,
        t_max = args.t_max,
        classes = classes[args.dataset],
        transform=utils.transforms[args.dataset]
    )

    # visualize some examples
    image_list = [utils.drawlines(train_data_ts.get_x_n(n), train_data_ts.splits[n][0][0])
    for n in range(0, 5)]
    images = torch.cat(image_list, dim=0)
    save_image(make_grid(images, nrow=train_data_ts.T), Path(args.output_dir, f"TS.png"))

    valid_data_ts = datasets.TS(
        datapath = args.datapath,
        dataset = args.dataset,
        split = "train",
        n_max = int(0.2*args.n_max),
        t_max = args.t_max,
        classes = classes[args.dataset],
        transform=utils.transforms[args.dataset]
    )

    train_queue_ts = DataLoader(
        train_data_ts,
        shuffle=False,
        num_workers=args.num_workers,
        batch_size = args.batch_size
    )
    valid_queue_ts = DataLoader(
        valid_data_ts,
        shuffle=False,
        num_workers=args.num_workers,
        batch_size=args.batch_size
    )

    model = networks.FCMLVAE(train_data_ts.dims, 500, args.cs_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    best_loss = float("inf")
    is_best = False
    start_epoch = 0
    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()

        logging.info(f"epoch {epoch}")

        train_s_kl, train_c_kl, train_recon, train_loss = train(
            train_queue_ts, model, optimizer, epoch, device
        )
        logging.info("train_s_kl %e train_c_kl %e train_recon %e train_loss %e",
                     train_s_kl, train_c_kl, train_recon, train_loss)
        with torch.no_grad():
            val_s_kl, val_c_kl, val_recon, val_loss = infer(
                valid_queue_ts, model, optimizer, epoch, device
            )
        if val_loss < best_loss:
            best_loss = val_loss
            is_best = True
        else:
            is_best = False
        logging.info("val_s_kl %e val_c_kl %e val_recon %e val_loss %e",
                     val_s_kl, val_c_kl, val_recon, val_loss)

        utils.save_checkpoint({
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }, is_best, args.output_dir)


def train(train_queue, model, optimizer, epoch, device):
    model.train()
    s_kl_meter = utils.AvgrageMeter()
    c_kl_meter = utils.AvgrageMeter()
    recon_loss_meter = utils.AvgrageMeter()
    loss_meter = utils.AvgrageMeter()
    

    for step, (input, target) in enumerate(train_queue):
        optimizer.zero_grad()
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        n = target.size(0)

        s_mu, s_logvar, c_mu, c_logvar = model(input)
        grouped_mu, grouped_logvar = utils.accumulate_group_evidence(c_mu, c_logvar, target, device)

        s_kl = torch.mean(-0.5 * torch.sum(1 + s_logvar - s_mu.pow(2) - s_logvar.exp(), dim=1), dim=0)
        c_kl = torch.mean(-0.5 * torch.sum(1 + grouped_logvar - grouped_mu.pow(2) - grouped_logvar.exp(), dim=1), dim=0)
        
        recon = utils.reconstruct(model, s_mu, s_logvar, grouped_mu, grouped_logvar, target, use_s=True)
        recon_loss = F.mse_loss(recon, input)

        loss = recon_loss + args.beta * (s_kl + args.w * c_kl)
        loss.backward()
        optimizer.step()

        s_kl_meter.update(s_kl.item(), n)
        c_kl_meter.update(c_kl.item(), n)
        recon_loss_meter.update(recon_loss.item(), n)
        loss_meter.update(loss.item(), n)

        if step % args.report_freq == 0:
            logging.info("train %03d s_kl %e c_kl %e recon %e loss %e", step, 
            s_kl_meter.avg, c_kl_meter.avg, recon_loss_meter.avg, loss_meter.avg)
            writer.add_scalar("s_kl_batch/train", s_kl_meter.avg, epoch * len(train_queue) + step)
            writer.add_scalar("c_kl_batch/train", c_kl_meter.avg, epoch * len(train_queue) + step)
            writer.add_scalar("recon_batch/train", recon_loss_meter.avg, epoch * len(train_queue) + step)
            writer.add_scalar("loss_batch/train", loss_meter.avg, epoch * len(train_queue) + step)
        
        writer.add_scalar("s_kl_epoch/train", s_kl_meter.avg, epoch)
        writer.add_scalar("c_kl_epoch/train", c_kl_meter.avg, epoch)
        writer.add_scalar("recon_epoch/train", recon_loss_meter.avg, epoch)
        writer.add_scalar("loss_epoch/train", loss_meter.avg, epoch)
    
    return s_kl_meter.avg, c_kl_meter.avg, recon_loss_meter.avg, loss_meter.avg


def infer(valid_queue, model, optimizer, epoch, device):
    model.eval()
    s_kl_meter = utils.AvgrageMeter()
    c_kl_meter = utils.AvgrageMeter()
    recon_loss_meter = utils.AvgrageMeter()
    loss_meter = utils.AvgrageMeter()
    

    for step, (input, target) in enumerate(valid_queue):
        optimizer.zero_grad()
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        n = target.size(0)

        s_mu, s_logvar, c_mu, c_logvar = model(input)
        grouped_mu, grouped_logvar = utils.accumulate_group_evidence(c_mu, c_logvar, target, device)

        s_kl = torch.mean(-0.5 * torch.sum(1 + s_logvar - s_mu.pow(2) - s_logvar.exp(), dim=1), dim=0)
        c_kl = torch.mean(-0.5 * torch.sum(1 + grouped_logvar - grouped_mu.pow(2) - grouped_logvar.exp(), dim=1), dim=0)

        recon = utils.reconstruct(model, s_mu, s_logvar, grouped_mu, grouped_logvar, target, use_s=True)
        recon_loss = F.mse_loss(recon, input)

        loss = recon_loss + args.beta * (s_kl + args.w * c_kl)

        s_kl_meter.update(s_kl.item(), n)
        c_kl_meter.update(c_kl.item(), n)
        recon_loss_meter.update(recon_loss.item(), n)
        loss_meter.update(loss.item(), n)

        if step % args.report_freq == 0:
            logging.info("valid %03d s_kl %e c_kl %e recon %e loss %e", step, 
            s_kl_meter.avg, c_kl_meter.avg, recon_loss_meter.avg, loss_meter.avg)
            writer.add_scalar("s_kl_batch/val", s_kl_meter.avg, epoch * len(valid_queue) + step)
            writer.add_scalar("c_kl_batch/val", c_kl_meter.avg, epoch * len(valid_queue) + step)
            writer.add_scalar("recon_batch/val", recon_loss_meter.avg, epoch * len(valid_queue) + step)
            writer.add_scalar("loss_batch/val", loss_meter.avg, epoch * len(valid_queue) + step)
        
        writer.add_scalar("s_kl_epoch/val", s_kl_meter.avg, epoch)
        writer.add_scalar("c_kl_epoch/val", c_kl_meter.avg, epoch)
        writer.add_scalar("recon_epoch/val", recon_loss_meter.avg, epoch)
        writer.add_scalar("loss_epoch/val", loss_meter.avg, epoch)
    
    return s_kl_meter.avg, c_kl_meter.avg, recon_loss_meter.avg, loss_meter.avg


if __name__ == "__main__":
    main()
