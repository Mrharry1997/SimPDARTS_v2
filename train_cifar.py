import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import genotypes
from genotypes import PRIMITIVES_INDEX
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
# from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from train_model import NetworkCIFAR as Network


parser = argparse.ArgumentParser("cifar_SimPDARTS_v2")
parser.add_argument('--workers', type=int, default=4, help='number of workers')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--adam_lr', type=float, default=0.0001, help='init Adam learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
parser.add_argument('--warmup_epochs', type=int, default=10, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='./experiments/', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='v2_test', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--tmp_data_dir', type=str, default='/home/harry/datasets', help='temp data dir')
parser.add_argument('--note', type=str, default='try', help='note for this run')
parser.add_argument('--cifar100', action='store_true', default=False, help='if use cifar100')
parser.add_argument('--load_weight', action='store_true', default=False, help='load weight to train')
parser.add_argument('--weight_path', type=str, default='./experiments/', help='the path of load weight')
parser.add_argument('--train_portion', type=float, default=1, help='portion of training data')

args, unparsed = parser.parse_known_args()

args.save = '{}eval-{}-{}'.format(args.save, args.note, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
# writer = SummaryWriter(args.save)

if args.cifar100:
    CIFAR_CLASSES = 100
    data_folder = 'cifar-100-python'
else:
    CIFAR_CLASSES = 10
    data_folder = 'cifar-10-batches-py'

def main():
    if not torch.cuda.is_available():
        logging.info('No GPU device available')
        sys.exit(1)
    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled=True
    torch.cuda.manual_seed(args.seed)
    logging.info("args = %s", args)
    logging.info("unparsed args = %s", unparsed)
    num_gpus = torch.cuda.device_count()
    
    architecture = eval("genotypes.%s" % args.arch)
    print('---------Genotype---------')
    logging.info(architecture)
    print('--------------------------')
    model = Network(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, architecture)
    model = torch.nn.DataParallel(model)
    model = model.cuda()
    # model_dict = model.state_dict()
    layer_number = [[[-1, -1] for _ in range(8)] for _ in range(len(architecture))]
    for i in range(len(architecture)):
        for j in range(len(architecture[i])):
            layer_number[i][j][0] = PRIMITIVES_INDEX[architecture[i][j][0]]
            if j < 2:
                layer_number[i][j][1] = architecture[i][j][1]
            elif j < 4:
                layer_number[i][j][1] = architecture[i][j][1] + 2
            elif j < 6:
                layer_number[i][j][1] = architecture[i][j][1] + 5
            else:
                layer_number[i][j][1] = architecture[i][j][1] + 9

    if args.load_weight:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(os.path.join(args.weight_path, 'finetune_weights.pt'))
        corrected_dict = {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and 'classifier' not in k:
                corrected_dict[k] = v
                continue
            k_split = k.split('.')
            if len(k_split) > 3:
                for i in range(8):
                    if ('cell_ops.' + str(layer_number[int(k_split[2])][i][1]) + '.') in k:
                        if layer_number[int(k_split[2])][i][0] == 1 or layer_number[int(k_split[2])][i][0] == 2:
                            continue
                        k_name = ''
                        for j in range(len(k_split)):
                            if j == 3:
                                k_name += '_ops'
                            elif j == 4:
                                k_name += str(i)
                            elif j == 5 or j == 6:
                                continue
                            else:
                                k_name += k_split[j]
                            if j != len(k_split) - 1: k_name += '.'
                        corrected_dict[k_name] = v
        # print(corrected_dict.keys())
        model_dict.update(corrected_dict)
        model.load_state_dict(model_dict)

        for name, param in model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    if args.load_weight:
        args.learning_rate = args.adam_lr
        optimizer_total = torch.optim.Adam(model.parameters(), args.learning_rate, betas=(0.9, 0.999), weight_decay=args.weight_decay)
        classifier_optimizer = torch.optim.Adam(model.module.classifier.parameters(), args.learning_rate, weight_decay=args.weight_decay)
        scheduler_warm = torch.optim.lr_scheduler.LambdaLR(classifier_optimizer, lr_lambda=lambda epoch:(epoch / args.warmup_epochs))
    else:
        optimizer_total = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
        args.warmup_epochs = 0

    if args.cifar100:
        train_transform, valid_transform = utils._data_transforms_cifar100(args)
    else:
        train_transform, valid_transform = utils._data_transforms_cifar10(args)
    if args.cifar100:
        train_data = dset.CIFAR100(root=args.tmp_data_dir, train=True, download=True, transform=train_transform)
        valid_data = dset.CIFAR100(root=args.tmp_data_dir, train=False, download=True, transform=valid_transform)
    else:
        train_data = dset.CIFAR10(root=args.tmp_data_dir, train=True, download=True, transform=train_transform)
        valid_data = dset.CIFAR10(root=args.tmp_data_dir, train=False, download=True, transform=valid_transform)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]), pin_memory=True, num_workers=args.workers)
    # train_queue = torch.utils.data.DataLoader(
    #     train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.workers)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_total, T_max=(args.epochs - args.warmup_epochs))

    best_acc = 0.0
    for epoch in range(args.epochs):
        if epoch == args.warmup_epochs:
            for param in model.parameters():
                param.requires_grad = True
        if epoch < args.warmup_epochs:
            optimizer = classifier_optimizer
            scheduler = scheduler_warm
        else:
            optimizer = optimizer_total
            scheduler = scheduler_cosine
        scheduler.step()
        logging.info('Epoch: %d lr %e', epoch, scheduler.get_lr()[0])
        model.module.drop_path_prob = args.drop_path_prob * epoch / args.epochs
        model.drop_path_prob = args.drop_path_prob * epoch / args.epochs
        start_time = time.time()
        train_acc, train_obj = train(train_queue, model, criterion, optimizer)
        logging.info('Train_acc: %f', train_acc)
        # writer.add_scalar('Train/accuracy', train_acc, epoch)
        # writer.add_scalar('Train/loss', train_obj, epoch)

        valid_acc, valid_obj = infer(valid_queue, model, criterion)
        # writer.add_scalar('Valid/accuracy', valid_acc, epoch)
        # writer.add_scalar('Valid/loss', valid_obj, epoch)
        if valid_acc > best_acc:
            best_acc = valid_acc
        logging.info('Valid_acc: %f', valid_acc)
        end_time = time.time()
        duration = end_time - start_time
        print('Epoch time: %ds.' % duration )
        utils.save(model.module, os.path.join(args.save, 'weights.pt'))

def train(train_queue, model, criterion, optimizer):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    model.train()

    for step, (input, target) in enumerate(train_queue):
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        optimizer.zero_grad()
        logits, logits_aux = model(input)
        loss = criterion(logits, target)
        if args.auxiliary:
            loss_aux = criterion(logits_aux, target)
            loss += args.auxiliary_weight*loss_aux
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, _ = utils.accuracy(logits, target, topk=(1,5))
        n = input.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)

        if step % args.report_freq == 0:
            logging.info('Train Step: %03d Objs: %e Acc: %f', step, objs.avg, top1.avg)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    model.eval()

    for step, (input, target) in enumerate(valid_queue):
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        with torch.no_grad():
            logits, _ = model(input)
            loss = criterion(logits, target)

        prec1, _ = utils.accuracy(logits, target, topk=(1,5))
        n = input.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)

        if step % args.report_freq == 0:
            logging.info('Valid Step: %03d Objs: %e Acc: %f', step, objs.avg, top1.avg)

    return top1.avg, objs.avg


if __name__ == '__main__':
    start_time = time.time()
    main() 
    end_time = time.time()
    duration = end_time - start_time
    logging.info('Eval time: %ds.', duration)
    
