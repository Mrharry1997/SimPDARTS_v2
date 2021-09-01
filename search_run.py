import argparse
import torch
import torch.backends.cudnn as cudnn
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset
import time
import sys
import os
import glob
import re
import numpy as np
import utils
import logging
import torch.nn as nn
import torch.utils
import copy
import torch.nn.functional as F
from model_search import Network
from genotypes import PRIMITIVES
from genotypes import Genotype
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

parser = argparse.ArgumentParser(description='SimCLR+PDARTS v2')
parser.add_argument('--data', metavar='DIR', default='/home/PJLAB/zhazhenzhou/Dataset/', help='path to dataset')
parser.add_argument('--dataset_name', default='cifar10', help='dataset name', choices=['cifar10', 'stl10', 'cifar100'])
parser.add_argument('--workers', type=int, default=8, help='number of workers to load dataset')
parser.add_argument('--epochs', default=25, type=int, metavar='N', help='number of epochs for the first stage to run')
parser.add_argument('--grow_epochs', default=6, type=int, metavar='N', help='number of epochs for train when the number of cells is appending (both for no_arch and arch)')
parser.add_argument('-b', '--batch-size', default=96, type=int, metavar='N', help='mini-batch size (default: 96), this is the total '
                                                                                    'batch size of all GPUs on the current node when '
                                                                                     'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.0, help='min learning rate')
parser.add_argument('--adam_lr', type=float, default=0.001, help='when load_weight is True, use Adam optimizer')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')     #same with train
parser.add_argument('--init_layers', type=int, default=5, help='init number of layers')
parser.add_argument('--total_layers', type=int, default=15, help='total number of layers')
parser.add_argument('--add_layer', type=int, default=1, help='number of layer that add each time')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='./experiments/', help='experiment path')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--arch_learning_rate', type=float, default=6e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--note', type=str, default='try', help='note for this run')
parser.add_argument('--dropout_rate', type=float, default=0.1, help='dropout rate of skip connect')
parser.add_argument('--fp16-precision', action='store_true', help='Whether or not to use 16-bit precision GPU training.')
parser.add_argument('--out_dim', default=128, type=int, help='feature dimension (default: 128)')
parser.add_argument('--temperature', default=0.07, type=float, help='softmax temperature (default: 0.07)')
parser.add_argument('--n-views', default=2, type=int, metavar='N', help='Number of views for contrastive learning training.')
parser.add_argument('--load_weight', type=bool, default=True, help='preserve weight for previous stage')

args = parser.parse_args()

args.save = '{}search-{}-{}'.format(args.save, args.note, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

# if args.dataset_name == 'cifar100':
#     CIFAR_CLASSES = 100
#     data_folder = 'cifar-100-python'
# elif args.dataset_name == 'cifar10':
#     CIFAR_CLASSES = 10
#     data_folder = 'cifar-10-batches-py'

def main():
    if not torch.cuda.is_available():
        logging.info('No GPU device available')
        sys.exit(1)
    args.device = torch.device('cuda')
    np.random.seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info("args = %s", args)

    dataset = ContrastiveLearningDataset(args.data)     #use SimCLR data augumentation
    train_dataset = dataset.get_dataset(args.dataset_name, args.n_views)
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    train_queue = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=args.workers)

    valid_queue = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True, num_workers=args.workers)

    # build Network
    criterion = nn.CrossEntropyLoss()  # change to SimCLR loss
    criterion = criterion.cuda()
    switches = []
    for i in range(14):
        switches.append([True for j in range(len(PRIMITIVES))])
    switches_normal = copy.deepcopy(switches)
    # switches_normal = [[False, True, True, False, False, True, False, False] for _ in range(14)]
    switches_reduce = copy.deepcopy(switches)

    num_to_keep = 1
    num_to_drop = 7

    drop_rate = args.dropout_rate

    layers = args.init_layers
    drop_used_rate = drop_rate

    scaler = GradScaler(enabled=args.fp16_precision)
    pre_layer = []

    model = Network(args.init_channels, args.out_dim, layers, pre_layer, args.init_layers, switches_normal=switches_normal, switches_reduce=switches_reduce, p=float(drop_used_rate))
    # layers += args.add_layer
    model = nn.DataParallel(model)
    model = model.cuda()
    model_dict = model.state_dict()
    network_params = []
    for k, v in model.named_parameters():
        if not (k.endswith('alphas_normal') or k.endswith('alphas_reduce')):
            network_params.append(v)
    optimizer = torch.optim.SGD(network_params, args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer_a = torch.optim.Adam(model.module.arch_parameters(), lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.learning_rate_min, last_epoch=-1)
    epochs = args.epochs
    eps_no_arch = 10
    scale_factor = 0.2
    for epoch in range(epochs):
        scheduler.step()
        lr = scheduler.get_lr()[0]
        logging.info('Epoch: %d lr: %e', epoch, lr)
        epoch_start = time.time()

        #training
        if epoch < eps_no_arch:
            model.module.p = float(drop_used_rate) * (epochs - epoch - 1) / epochs
            model.module.update_p()
            train_acc, train_obj = train(train_queue, valid_queue, model, network_params, criterion, optimizer, optimizer_a, scaler, train_arch=False)
        else:
            model.module.p = float(drop_used_rate) * np.exp(-(epoch - eps_no_arch) * scale_factor)
            model.module.update_p()
            train_acc, train_obj = train(train_queue, valid_queue, model, network_params, criterion, optimizer, optimizer_a, scaler, train_arch=True)

        logging.info('Train_acc %f', train_acc)
        epoch_duration = time.time() - epoch_start
        logging.info('Epoch time: %ds', epoch_duration)

    utils.save(model, os.path.join(args.save, 'weights.pt'))
    print('------Dropping %d paths------' % num_to_drop)
    print('------Dropping the first stage------')
    # Save switches info for s-c refinement.
    switches_normal_2 = copy.deepcopy(switches_normal)
    switches_reduce_2 = copy.deepcopy(switches_reduce)
    # drop operations with low architecture weights
    arch_param = model.module.arch_parameters()
    normal_prob = F.softmax(arch_param[0], dim=-1).data.cpu().numpy()
    for i in range(14):
        idxs = []
        for j in range(len(PRIMITIVES)):
            if switches_normal[i][j]:
                idxs.append(j)
            # drop all Zero operations
        drop = get_min_k_no_zero(normal_prob[i, :], idxs, num_to_drop)
        for idx in drop:
            switches_normal[i][idxs[idx]] = False
    reduce_prob = F.softmax(arch_param[1], dim=-1).data.cpu().numpy()
    for i in range(14):
        idxs = []
        for j in range(len(PRIMITIVES)):
            if switches_reduce[i][j]:
                idxs.append(j)
        drop = get_min_k_no_zero(reduce_prob[i, :], idxs, num_to_drop)
        for idx in drop:
            switches_reduce[i][idxs[idx]] = False
    logging.info('switches_normal = %s', switches_normal)
    logging_switches(switches_normal)
    logging.info('switches_reduce = %s', switches_reduce)
    logging_switches(switches_reduce)

    # restrict skip-connection
    normal_final = [0 for idx in range(14)]
    reduce_final = [0 for idx in range(14)]
    # remove all Zero operations
    for i in range(14):
        if switches_normal_2[i][0] == True:
            normal_prob[i][0] = 0
        normal_final[i] = max(normal_prob[i])
        if switches_reduce_2[i][0] == True:
            reduce_prob[i][0] = 0
        reduce_final[i] = max(reduce_prob[i])
        # Generate Architecture, similar to DARTS
    keep_normal = [0, 1]
    keep_reduce = [0, 1]
    n = 3
    start = 2
    for i in range(3):
        end = start + n
        tbsn = normal_final[start:end]
        tbsr = reduce_final[start:end]
        edge_n = sorted(range(n), key=lambda x: tbsn[x])
        keep_normal.append(edge_n[-1] + start)
        keep_normal.append(edge_n[-2] + start)
        edge_r = sorted(range(n), key=lambda x: tbsr[x])
        keep_reduce.append(edge_r[-1] + start)
        keep_reduce.append(edge_r[-2] + start)
        start = end
        n = n + 1
    # set switches according the ranking of arch parameters
    for i in range(14):
        if not i in keep_normal:
            for j in range(len(PRIMITIVES)):
                switches_normal[i][j] = False
        if not i in keep_reduce:
            for j in range(len(PRIMITIVES)):
                switches_reduce[i][j] = False
    # translate switches into genotype
    genotype = parse_network(switches_normal, switches_reduce)
    logging.info(genotype)
    ## restrict skipconnect (normal cell only)
    logging.info('Restricting skipconnect...')
    # generating genotypes with different numbers of skip-connect operations
    switches_usable = False
    for sks in range(0, 9):
        max_sk = 8 - sks
        num_sk = check_sk_number(switches_normal)
        if not num_sk > max_sk:
            continue
        while num_sk > max_sk:
            normal_prob = delete_min_sk_prob(switches_normal, switches_normal_2, normal_prob)
            switches_normal = keep_1_on(switches_normal_2, normal_prob)
            switches_normal = keep_2_branches(switches_normal, normal_prob)
            num_sk = check_sk_number(switches_normal)
        logging.info('Number of skip-connect: %d', max_sk)
        genotype = parse_network(switches_normal, switches_reduce)
        logging.info(genotype)

        if not switches_usable and max_sk <= 2:
            switches_normal_usable = switches_normal
            switches_reduce_usable = switches_reduce
            logging.info('usable_switches_normal = %s', switches_normal_usable)
            logging.info('usable_switches_reduce = %s', switches_reduce_usable)
            switches_usable = True
    if not switches_usable:
        switches_normal_usable = switches_normal
        switches_reduce_usable = switches_reduce
        logging.info('usable_switches_normal = %s', switches_normal_usable)
        logging.info('usable_switches_reduce = %s', switches_reduce_usable)

    for i in range(layers):
        if i == 4 or i == 9:
            pre_layer.append(switches_reduce_usable)
        else:
            pre_layer.append(switches_normal_usable)

    utils.save(model, os.path.join(args.save, 'weights.pt'))

    # switches_reduce_usable = [[False, False, False, False, False, False, True, False], [False, False, False, False, False, True, False, False], [False, False, False, False, False, False, False, True], [False, True, False, False, False, False, False, False], [False, False, False, False, False, False, False, False], [False, False, False, False, False, False, True, False], [False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False], [False, False, False, False, True, False, False, False], [False, False, False, False, False, False, False, False], [False, False, False, True, False, False, False, False], [False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False], [False, False, False, False, False, True, False, False]]
    # switches_normal_usable = [[False, False, False, True, False, False, False, False], [False, False, False, True, False, False, False, False], [False, True, False, False, False, False, False, False], [False, False, False, False, False, False, False, True], [False, False, False, False, False, False, False, False], [False, False, False, False, True, False, False, False], [False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, True], [False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False], [False, False, True, False, False, False, False, False], [False, True, False, False, False, False, False, False]]
    #
    #
    # for i in range(layers):
    #     if i == 4 or i == 9:
    #         pre_layer.append(switches_reduce_usable)
    #     else:
    #         pre_layer.append(switches_normal_usable)

    while layers <= args.total_layers:
        switches_normal = copy.deepcopy(switches)
        switches_reduce = copy.deepcopy(switches)
        layers += args.add_layer
        model = Network(args.init_channels, args.out_dim, layers, pre_layer, args.init_layers, switches_normal=switches_normal, switches_reduce=switches_reduce, p=float(drop_used_rate))
        model = nn.DataParallel(model)
        model = model.cuda()
        normal_number = [-1] * len(switches_normal_usable)
        reduce_number = [-1] * len(switches_reduce_usable)
        for i in range(len(switches_normal_usable)):
            for j in range(len(switches_normal_usable[i])):
                if switches_normal_usable[i][j]:
                    normal_number[i] = j
        for i in range(len(switches_reduce_usable)):
            for j in range(len(switches_reduce_usable[i])):
                if switches_reduce_usable[i][j]:
                    reduce_number[i] = j
        if args.load_weight:
            model_dict = model.state_dict()
            # pretrained_dict = torch.load(os.path.join(args.save, 'weights.pt'))
            pretrained_dict = torch.load('./experiments/search-try-20210901-100834/weights.pt')
            corrected_dict = {}
            for k, v in pretrained_dict.items():
                if 'mlp' in k:
                    continue
                if k in model_dict.keys():
                    corrected_dict[k] = v
                k_split = k.split('.')
                if len(k_split) > 3 and ( layers - args.add_layer == args.init_layers or int(k_split[2]) > layers - 2*args.add_layer - 1):
                    if int(k_split[2]) != 4 and int(k_split[2]) != 9:
                        for i in range(len(normal_number)):
                            if ('m_ops.'+str(normal_number[i])+'.') in k and ('cell_ops.'+str(i)+'.') in k:
                                k_name = ''
                                for j in range(len(k_split)):
                                    if j == 6:
                                        k_name += '0'
                                    else:
                                        k_name += k_split[j]
                                    if j != len(k_split)-1: k_name += '.'
                                corrected_dict[k_name] = v
                    else:
                        for i in range(len(reduce_number)):
                            if ('m_ops.'+str(reduce_number[i])+'.') in k and ('cell_ops.'+str(i)+'.') in k:
                                k_name = ''
                                for j in range(len(k_split)):
                                    if j == 6:
                                        k_name += '0'
                                    else:
                                        k_name += k_split[j]
                                    if j != len(k_split)-1: k_name += '.'
                                corrected_dict[k_name] = v
            print(corrected_dict.keys())
            model_dict.update(corrected_dict)
            model.load_state_dict(model_dict)
        network_params = []
        arch_parameter = []
        for k, v in model.named_parameters():
            if not (k.endswith('alphas_normal') or k.endswith('alphas_reduce')):
                network_params.append(v)
        if layers == 5 or layers == 10:
            arch_parameter.append(model.module.arch_parameters()[1])
        elif layers != 5 and layers != 10:
            arch_parameter.append(model.module.arch_parameters()[0])
        if args.load_weight:
            optimizer = torch.optim.Adam(network_params, args.adam_lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.SGD(network_params, args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
        optimizer_a = torch.optim.Adam(arch_parameter, lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.grow_epochs, eta_min=args.learning_rate_min, last_epoch=-1)
        sm_dim = -1
        epochs = args.grow_epochs
        eps_no_arch = args.grow_epochs//2
        scale_factor = 0.2
        for epoch in range(epochs):
            # if load_weight is True then warmup for the first 5 epochs
            if not args.load_weight:
                scheduler.step()
            elif epoch >= eps_no_arch:
                scheduler.step()

            lr = scheduler.get_lr()[0]
            logging.info('Epoch: %d lr: %e', epoch, lr)
            epoch_start = time.time()

            # training
            if epoch < eps_no_arch:
                model.module.p = float(drop_used_rate) * (epochs - epoch - 1) / epochs
                model.module.update_p()
                train_acc, train_obj = train(train_queue, valid_queue, model, network_params, criterion, optimizer, optimizer_a, scaler, train_arch=False)
            else:
                model.module.p = float(drop_used_rate) * np.exp(-(epoch - eps_no_arch) * scale_factor)
                model.module.update_p()
                train_acc, train_obj = train(train_queue, valid_queue, model, network_params, criterion, optimizer, optimizer_a, scaler, train_arch=True)

            logging.info('Train_acc %f', train_acc)
            epoch_duration = time.time() - epoch_start
            logging.info('Epoch time: %ds', epoch_duration)

        utils.save(model, os.path.join(args.save, 'weights.pt'))
        print('------Dropping %d paths------' % num_to_drop)
        # Save switches info for s-c refinement.
        switches_normal_2 = copy.deepcopy(switches_normal)
        switches_reduce_2 = copy.deepcopy(switches_reduce)
        # drop operations with low architecture weights
        arch_param = model.module.arch_parameters()
        normal_prob = F.softmax(arch_param[0], dim=sm_dim).data.cpu().numpy()
        for i in range(14):
            idxs = []
            for j in range(len(PRIMITIVES)):
                if switches_normal[i][j]:
                    idxs.append(j)
                # drop all Zero operations
            drop = get_min_k_no_zero(normal_prob[i, :], idxs, num_to_drop)
            for idx in drop:
                switches_normal[i][idxs[idx]] = False
        reduce_prob = F.softmax(arch_param[1], dim=-1).data.cpu().numpy()
        for i in range(14):
            idxs = []
            for j in range(len(PRIMITIVES)):
                if switches_reduce[i][j]:
                    idxs.append(j)
            drop = get_min_k_no_zero(reduce_prob[i, :], idxs, num_to_drop)
            for idx in drop:
                switches_reduce[i][idxs[idx]] = False
        if layers == 5 or layers == 10:
            logging.info('switches_reduce = %s', switches_reduce)
            logging_switches(switches_reduce)
        else:
            logging.info('switches_normal = %s', switches_normal)
            logging_switches(switches_normal)


        arch_param = model.module.arch_parameters()
        normal_prob = F.softmax(arch_param[0], dim=sm_dim).data.cpu().numpy()
        reduce_prob = F.softmax(arch_param[1], dim=sm_dim).data.cpu().numpy()
        normal_final = [0 for idx in range(14)]
        reduce_final = [0 for idx in range(14)]
        # remove all Zero operations
        for i in range(14):
            if switches_normal_2[i][0] == True:
                normal_prob[i][0] = 0
            normal_final[i] = max(normal_prob[i])
            if switches_reduce_2[i][0] == True:
                reduce_prob[i][0] = 0
            reduce_final[i] = max(reduce_prob[i])
            # Generate Architecture, similar to DARTS
        keep_normal = [0, 1]
        keep_reduce = [0, 1]
        n = 3
        start = 2
        for i in range(3):
            end = start + n
            tbsn = normal_final[start:end]
            tbsr = reduce_final[start:end]
            edge_n = sorted(range(n), key=lambda x: tbsn[x])
            keep_normal.append(edge_n[-1] + start)
            keep_normal.append(edge_n[-2] + start)
            edge_r = sorted(range(n), key=lambda x: tbsr[x])
            keep_reduce.append(edge_r[-1] + start)
            keep_reduce.append(edge_r[-2] + start)
            start = end
            n = n + 1
        # set switches according the ranking of arch parameters
        for i in range(14):
            if not i in keep_normal:
                for j in range(len(PRIMITIVES)):
                    switches_normal[i][j] = False
            if not i in keep_reduce:
                for j in range(len(PRIMITIVES)):
                    switches_reduce[i][j] = False
        # translate switches into genotype
        genotype = parse_network(switches_normal, switches_reduce)
        logging.info(genotype)
        ## restrict skipconnect (normal cell only)
        logging.info('Restricting skipconnect...')
        # generating genotypes with different numbers of skip-connect operations
        switches_usable = False
        for sks in range(0, 9):
            max_sk = 8 - sks
            num_sk = check_sk_number(switches_normal)
            if not num_sk > max_sk:
                continue
            while num_sk > max_sk:
                normal_prob = delete_min_sk_prob(switches_normal, switches_normal_2, normal_prob)
                switches_normal = keep_1_on(switches_normal_2, normal_prob)
                switches_normal = keep_2_branches(switches_normal, normal_prob)
                num_sk = check_sk_number(switches_normal)
            logging.info('Number of skip-connect: %d', max_sk)
            genotype = parse_network(switches_normal, switches_reduce)
            logging.info(genotype)

            if not switches_usable and max_sk <= 2:
                switches_normal_usable = switches_normal
                switches_reduce_usable = switches_reduce
                switches_usable = True
        if not switches_usable:
            switches_normal_usable = switches_normal
            switches_reduce_usable = switches_reduce

        if layers == 5 or layers == 10:
            logging.info('usable_switches_reduce = %s', switches_reduce_usable)
            pre_layer.append(switches_reduce_usable)
        else:
            logging.info('usable_switches_normal = %s', switches_normal_usable)
            pre_layer.append(switches_normal_usable)

        logging.info('The %d layers is finished'%layers)
        utils.save(model, os.path.join(args.save, 'weights.pt'))

    logging.info('Search process is finished')
    final_arch = []
    for switches_final in pre_layer:
        final_arch.append(parse_switches(switches_final))
    logging.info(final_arch)



def parse_network(switches_normal, switches_reduce):
    def _parse_switches(switches):
        n = 2
        start = 0
        gene = []
        step = 4
        for i in range(step):
            end = start + n
            for j in range(start, end):
                for k in range(len(switches[j])):
                    if switches[j][k]:
                        gene.append((PRIMITIVES[k], j - start))
            start = end
            n = n + 1
        return gene

    gene_normal = _parse_switches(switches_normal)
    gene_reduce = _parse_switches(switches_reduce)

    concat = range(2, 6)

    genotype = Genotype(
        normal=gene_normal, normal_concat=concat,
        reduce=gene_reduce, reduce_concat=concat
    )

    return genotype

def parse_switches(switches):
    n = 2
    start = 0
    gene = []
    step = 4
    for i in range(step):
        end = start + n
        for j in range(start, end):
            for k in range(len(switches[j])):
                if switches[j][k]:
                    gene.append((PRIMITIVES[k], j - start))
        start = end
        n = n + 1
    return gene

def get_min_k(input_in, k):
    input = copy.deepcopy(input_in)
    index = []
    for i in range(k):
        idx = np.argmin(input)
        index.append(idx)
        input[idx] = 1

    return index


def get_min_k_no_zero(w_in, idxs, k):
    w = copy.deepcopy(w_in)
    index = []
    if 0 in idxs:
        zf = True
    else:
        zf = False
    if zf:
        w = w[1:]
        index.append(0)
        k = k - 1
    for i in range(k):
        idx = np.argmin(w)
        w[idx] = 1
        if zf:
            idx = idx + 1
        index.append(idx)
    return index


def logging_switches(switches):
    for i in range(len(switches)):
        ops = []
        for j in range(len(switches[i])):
            if switches[i][j]:
                ops.append(PRIMITIVES[j])
        logging.info(ops)


def check_sk_number(switches):
    count = 0
    for i in range(len(switches)):
        if switches[i][3]:
            count = count + 1

    return count


def delete_min_sk_prob(switches_in, switches_bk, probs_in):
    def _get_sk_idx(switches_in, switches_bk, k):
        if not switches_in[k][3]:
            idx = -1
        else:
            idx = 0
            for i in range(3):
                if switches_bk[k][i]:
                    idx = idx + 1
        return idx

    probs_out = copy.deepcopy(probs_in)
    sk_prob = [1.0 for i in range(len(switches_bk))]
    for i in range(len(switches_in)):
        idx = _get_sk_idx(switches_in, switches_bk, i)
        if not idx == -1:
            sk_prob[i] = probs_out[i][idx]
    d_idx = np.argmin(sk_prob)
    idx = _get_sk_idx(switches_in, switches_bk, d_idx)
    probs_out[d_idx][idx] = 0.0

    return probs_out


def keep_1_on(switches_in, probs):
    switches = copy.deepcopy(switches_in)
    for i in range(len(switches)):
        idxs = []
        for j in range(len(PRIMITIVES)):
            if switches[i][j]:
                idxs.append(j)
        drop = get_min_k_no_zero(probs[i, :], idxs, 7)
        for idx in drop:
            switches[i][idxs[idx]] = False
    return switches


def keep_2_branches(switches_in, probs):
    switches = copy.deepcopy(switches_in)
    final_prob = [0.0 for i in range(len(switches))]
    for i in range(len(switches)):
        final_prob[i] = max(probs[i])
    keep = [0, 1]
    n = 3
    start = 2
    for i in range(3):
        end = start + n
        tb = final_prob[start:end]
        edge = sorted(range(n), key=lambda x: tb[x])
        keep.append(edge[-1] + start)
        keep.append(edge[-2] + start)
        start = end
        n = n + 1
    for i in range(len(switches)):
        if not i in keep:
            for j in range(len(PRIMITIVES)):
                switches[i][j] = False
    return switches




def info_nce_loss(features):
    labels = torch.cat([torch.arange(features.shape[0]//2) for i in range(args.n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(args.device)

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)
    # assert similarity_matrix.shape == (
    #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
    # assert similarity_matrix.shape == labels.shape

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(args.device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(args.device)

    logits = logits / args.temperature
    return logits, labels


def train(train_queue, valid_queue, model, network_params, criterion, optimizer, optimizer_a, scaler, train_arch=True):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    n_iter = 0

    for images, _ in tqdm(train_queue):
        model.train()
        n = images[0].size(0)
        images = torch.cat(images, dim=0)
        images = images.cuda()

        if train_arch:
            try:
                images_search, _ = next(valid_queue_iter)
            except:
                valid_queue_iter = iter(valid_queue)
                images_search, _ = next(valid_queue_iter)
            images_search = torch.cat(images_search, dim=0)
            images_search = images_search.cuda()
            optimizer_a.zero_grad()
            with autocast(enabled=args.fp16_precision):
                features_a = model(images_search)
                logits_a, labels_a = info_nce_loss(features_a)
                loss_a = criterion(logits_a, labels_a)
            scaler.scale(loss_a).backward()
            nn.utils.clip_grad_norm_(model.module.arch_parameters(), args.grad_clip)
            scaler.step(optimizer_a)

        optimizer.zero_grad()
        with autocast(enabled=args.fp16_precision):
            features = model(images)
            logits, labels = info_nce_loss(features)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        nn.utils.clip_grad_norm_(network_params, args.grad_clip)
        scaler.step(optimizer)

        prec1, prec5 = utils.accuracy(logits, labels, topk=(1, 5))
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if n_iter % args.report_freq == 0:
            logging.info('TRAIN Step: %03d Objs: %e R1: %f R5: %f', n_iter, objs.avg, top1.avg, top5.avg)

        n_iter += 1

    return top1.avg, objs.avg

if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    duration = end_time - start_time
    logging.info('Total searching time: %ds', duration)