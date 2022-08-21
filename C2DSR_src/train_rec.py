import os
import sys
from datetime import datetime
import time
import numpy as np
import random
import argparse
from shutil import copyfile
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from utils import torch_utils, helper
from utils.GraphMaker import GraphMaker
from model.trainer import CDSRTrainer
from utils.loader import *
import json
import codecs
import tqdm
import pdb


parser = argparse.ArgumentParser()
# dataset part
parser.add_argument('--data_dir', type=str, default='Food-Kitchen', help='Movie-Book, Entertainment-Education')

# model part
parser.add_argument('--model', type=str, default="C2DSR", help='model name')
parser.add_argument('--hidden_units', type=int, default=256, help='lantent dim.')
parser.add_argument('--num_blocks', type=int, default=2, help='lantent dim.')
parser.add_argument('--num_heads', type=int, default=1, help='lantent dim.')
parser.add_argument('--GNN', type=int, default=1, help='GNN depth.')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate.')
parser.add_argument('--optim', choices=['sgd', 'adagrad', 'adam', 'adamax'], default='adam',
                    help='Optimizer: sgd, adagrad, adam or adamax.')
parser.add_argument('--lr', type=float, default=0.001, help='Applies to sgd and adagrad.')
parser.add_argument('--lr_decay', type=float, default=1, help='Learning rate decay rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--decay_epoch', type=int, default=5, help='Decay learning rate after this epoch.')
parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')
parser.add_argument('--leakey', type=float, default=0.1)
parser.add_argument('--maxlen', type=int, default=15)
parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--lambda', type=float, default=0.7)

# train part
parser.add_argument('--num_epoch', type=int, default=100, help='Number of total training epochs.')
parser.add_argument('--batch_size', type=int, default=256, help='Training batch size.')
parser.add_argument('--log_step', type=int, default=200, help='Print log every k steps.')
parser.add_argument('--log', type=str, default='logs.txt', help='Write training log to file.')
parser.add_argument('--save_epoch', type=int, default=100, help='Save model checkpoints every k epochs.')
parser.add_argument('--save_dir', type=str, default='./saved_models', help='Root dir for saving models.')
parser.add_argument('--id', type=str, default='00', help='Model ID under which to save models.')
parser.add_argument('--seed', type=int, default=2040)
parser.add_argument('--load', dest='load', action='store_true', default=False,  help='Load pretrained model.')
parser.add_argument('--model_file', type=str, help='Filename of the pretrained model.')
parser.add_argument('--info', type=str, default='', help='Optional info for the experiment.')
parser.add_argument('--undebug', action='store_false', default=True)

def seed_everything(seed=1111):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

args = parser.parse_args()
if args.cpu:
    args.cuda = False
elif args.cuda:
    torch.cuda.manual_seed(args.seed)
init_time = time.time()
# make opt
opt = vars(args)

seed_everything(opt["seed"])

model_id = opt['id'] if len(opt['id']) > 1 else '0' + opt['id']
model_save_dir = opt['save_dir'] + '/' + model_id
opt['model_save_dir'] = model_save_dir
helper.ensure_dir(model_save_dir, verbose=True)
# save config
helper.save_config(opt, model_save_dir + '/config.json', verbose=True)
file_logger = helper.FileLogger(model_save_dir + '/' + opt['log'],
                                header="# epoch\ttrain_loss\tdev_loss\tdev_score\tbest_dev_score")

# print model info
helper.print_config(opt)

if opt["undebug"]:
    pass
    # opt["cuda"] = False
    # opt["cpu"] = True

print("Loading data from {} with batch size {}...".format(opt['data_dir'], opt['batch_size']))
train_batch = DataLoader(opt['data_dir'], opt['batch_size'], opt, evaluation = -1)
valid_batch = DataLoader(opt['data_dir'], opt["batch_size"], opt, evaluation = 2)
test_batch = DataLoader(opt['data_dir'], opt["batch_size"], opt, evaluation = 1)
print("Data loading done!")

opt["itemnum"] = opt["source_item_num"] + opt["target_item_num"] + 1


filename = opt["data_dir"]
train_data = "./dataset/" + filename + "/traindata_new.txt"
G = GraphMaker(opt, train_data)
adj, adj_single = G.adj, G.adj_single
print("graph loaded!")

if opt["cuda"]:
    adj = adj.cuda()
    adj_single = adj_single.cuda()



# model
if not opt['load']:
    trainer = CDSRTrainer(opt, adj, adj_single)
else:
    exit(0)

global_step = 0
current_lr = opt["lr"]
format_str = '{}: step {}/{} (epoch {}/{}), loss = {:.6f} ({:.3f} sec/epoch), lr: {:.6f}'
num_batch = len(train_batch)
max_steps = opt['num_epoch'] * num_batch

print("Start training:")

begin_time = time.time()
X_dev_score_history=[0]
Y_dev_score_history=[0]
# start training
for epoch in range(1, opt['num_epoch'] + 1):
    train_loss = 0
    epoch_start_time = time.time()
    trainer.mi_loss = 0
    for batch in train_batch:
        global_step += 1
        loss = trainer.train_batch(batch)
        train_loss += loss

    duration = time.time() - epoch_start_time
    print(format_str.format(datetime.now(), global_step, max_steps, epoch, \
                                    opt['num_epoch'], train_loss/num_batch, duration, current_lr))
    print("mi:", trainer.mi_loss/num_batch)

    if epoch % 5:
        continue

    # eval model
    print("Evaluating on dev set...")

    trainer.model.eval()
    trainer.model.graph_convolution()


    def cal_test_score(predictions):
        MRR=0.0
        HR_1 = 0.0
        HR_5 = 0.0
        HR_10 = 0.0
        NDCG_5 = 0.0
        NDCG_10 = 0.0
        valid_entity = 0.0
        # pdb.set_trace()
        for pred in predictions:
            valid_entity += 1
            MRR += 1 / pred
            if pred <= 1:
                HR_1 += 1
            if pred <= 5:
                NDCG_5 += 1 / np.log2(pred + 1)
                HR_5 += 1
            if pred <= 10:
                NDCG_10 += 1 / np.log2(pred + 1)
                HR_10 += 1
            if valid_entity % 100 == 0:
                print('.', end='')
        return MRR/valid_entity, NDCG_5 / valid_entity, NDCG_10 / valid_entity, HR_1 / valid_entity, HR_5 / valid_entity, HR_10 / valid_entity

    def get_evaluation_result(evaluation_batch):
        X_pred = []
        Y_pred = []
        for i, batch in enumerate(evaluation_batch):
            X_predictions, Y_predictions = trainer.test_batch(batch)
            X_pred = X_pred + X_predictions
            Y_pred = Y_pred + Y_predictions

        return X_pred, Y_pred


    val_X_pred, val_Y_pred = get_evaluation_result(valid_batch)
    val_X_MRR, val_X_NDCG_5, val_X_NDCG_10, val_X_HR_1, val_X_HR_5, val_X_HR_10 = cal_test_score(val_X_pred)
    val_Y_MRR, val_Y_NDCG_5, val_Y_NDCG_10, val_Y_HR_1, val_Y_HR_5, val_Y_HR_10 = cal_test_score(val_Y_pred)

    print("")
    print('val epoch:%d, time: %f(s), X (MRR: %.4f, NDCG@10: %.4f, HR@10: %.4f), Y (MRR: %.4f, NDCG@10: %.4f, HR@10: %.4f)'
          % (epoch, time.time() - begin_time, val_X_MRR, val_X_NDCG_10, val_X_HR_10, val_Y_MRR, val_Y_NDCG_10, val_Y_HR_10))

    if val_X_MRR > max(X_dev_score_history) or val_Y_MRR > max(Y_dev_score_history):
        test_X_pred, test_Y_pred = get_evaluation_result(test_batch)
        test_X_MRR, test_X_NDCG_5, test_X_NDCG_10, test_X_HR_1, test_X_HR_5, test_X_HR_10 = cal_test_score(test_X_pred)
        test_Y_MRR, test_Y_NDCG_5, test_Y_NDCG_10, test_Y_HR_1, test_Y_HR_5, test_Y_HR_10 = cal_test_score(test_Y_pred)

        print("")
        if val_X_MRR > max(X_dev_score_history):
            print("X best!")
            print([test_X_MRR, test_X_NDCG_5, test_X_NDCG_10, test_X_HR_1, test_X_HR_5, test_X_HR_10])

        if val_Y_MRR > max(Y_dev_score_history):
            print("Y best!")
            print([test_Y_MRR, test_Y_NDCG_5, test_Y_NDCG_10, test_Y_HR_1, test_Y_HR_5, test_Y_HR_10])

    X_dev_score_history.append(val_X_MRR)
    Y_dev_score_history.append(val_Y_MRR)
