# from __future__ import annotations
import copy
import gc
import sys
import time
import torch
import numpy as np
from tqdm import tqdm
from statistics import mean
#from torchinfo import summary
from collections import OrderedDict
from itertools import islice
from numpy.random import randint
from configurations import args_parser
import utils
import models

# https://pypi.tuna.tsinghua.edu.cn/simple/   # 清华
# http://pypi.doubanio.com/simple/  # 豆瓣
# http://mirrors.aliyun.com/pypi/simple/  # 阿里云
# http://pypi.hustunique.com/  # 华中理工大学
# http://pypi.sdutlinux.org/ # 山东理工大学
# https://pypi.mirrors.ustc.edu.cn/simple/ # 中国科学技术大学
# http://mirrors.163.com/pypi/simple/ # 网易
if __name__ == '__main__':
    start_time = time.time()
    args = args_parser()
    boardio, textio, best_val_acc, path_best_model = utils.initializations(args)
    textio.cprint(str(args))

    # 给写入的文件命名
    if args.stragglers is not None:
        filename = "./records/Epoch"+str(args.global_epochs)+"CNN_batch" + str(args.train_batch_size) + "stragglers_percent" + str(args.stragglers_percent * args.num_users) + ".csv"
        timeFilename = "./records/Epoch"+str(args.global_epochs)+"TimeCNN_batch" + str(args.train_batch_size) + "stragglers_percent" + str(args.stragglers_percent * args.num_users) + ".csv"
    else:
        filename = "./records/Epoch"+str(args.global_epochs)+"CNN_batch" + str(args.train_batch_size) + "stragglers_percentNo" + ".csv"
        timeFilename = "./records/Epoch"+str(args.global_epochs)+"TimeCNN_batch" + str(args.train_batch_size) + "stragglers_percentNo" + ".csv"
    # 记录准确度\
    records = []
    timeRecords = []

    # data
    train_data, test_loader = utils.data(args)
    input, output, train_data, val_loader = utils.data_split(train_data, len(test_loader.dataset), args)

    # model
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if args.model == 'mlp':
        global_model = models.FC2Layer(input, output)
    else:
        global_model = models.CNN2Layer(input, output, args.data)
    #textio.cprint(str(summary(global_model)))
    global_model.to(args.device)
    # global_model.to(dev)

    train_creterion = torch.nn.CrossEntropyLoss(reduction='mean')
    test_creterion = torch.nn.CrossEntropyLoss(reduction='sum')

    # learning curve
    train_loss_list = []
    val_acc_list = []

    #  inference
    if args.eval:
        global_model.load_state_dict(torch.load(path_best_model))
        test_acc = utils.test(test_loader, global_model, test_creterion, args.device)
        textio.cprint(f'eval test_acc: {test_acc:.0f}%')
        gc.collect()
        sys.exit()

    local_models = utils.federated_setup(global_model, train_data, args)

    # stragglers
    num_of_layers = global_model.state_dict().keys().__len__()
    if args.stragglers is not None:
        stragglers_idx = randint(low=0, high=args.num_users,
                                 size=round(args.stragglers_percent * args.num_users))  # randomly choose the stragglers
    else:
        stragglers_idx = []

    for global_epoch in tqdm(range(0, args.global_epochs)):
        utils.distribute_model(local_models, global_model)
        users_loss = []

        for user_idx in range(args.num_users):
            if (args.stragglers == 'drop') & (user_idx in stragglers_idx):
                continue

            user_loss = []
            for local_epoch in range(0, args.local_epochs):
                user = local_models[user_idx]
                train_loss = utils.train_one_epoch(user['data'], user['model'], user['opt'],
                                                   train_creterion, args.device, args.local_iterations)
                user_loss.append(train_loss)

            if (args.stragglers == 'salf') & (user_idx in stragglers_idx):
                user_new_state_dict = copy.deepcopy(global_model).state_dict()
                if args.up_to_layer is not None:
                    up_to_layer = num_of_layers - args.up_to_layer  # last-to-first layers updated
                else:
                    up_to_layer = np.random.randint(1, num_of_layers + 1)  # random last-to-first layers updated

                user_updated_layers = OrderedDict(islice(reversed(user['model'].state_dict().items()), up_to_layer))
                user_new_state_dict.update(user_updated_layers)
                user['model'].load_state_dict(user_new_state_dict)

            users_loss.append(mean(user_loss))

        train_loss = mean(users_loss)
        utils.FedAvg(local_models, global_model)

        val_acc = utils.test(val_loader, global_model, test_creterion, args.device)
        train_loss_list.append(train_loss)
        val_acc_list.append(val_acc)

        boardio.add_scalar('train', train_loss, global_epoch)
        boardio.add_scalar('validation', val_acc, global_epoch)
        gc.collect()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(global_model.state_dict(), path_best_model)

        test_acc = utils.test(test_loader, global_model, test_creterion, args.device)
        textio.cprint(f'epoch: {global_epoch} | train_loss: {train_loss:.2f} | val_acc: {val_acc:.0f}%')
        records.append(val_acc)
        timeSub = time.time() - start_time
        timeSub = int(timeSub)
        timeRecords.append(timeSub)

    np.save(f'checkpoints/{args.exp_name}/train_loss_list.npy', train_loss_list)
    np.save(f'checkpoints/{args.exp_name}/val_acc_list.npy', val_acc_list)

    elapsed_min = (time.time() - start_time) / 60
    textio.cprint(f'total execution time: {elapsed_min:.1f} min')
    # 将结果写入文件
    utils.write_list_to_csv(filename, records)
    utils.write_list_to_csv(timeFilename, timeRecords)
