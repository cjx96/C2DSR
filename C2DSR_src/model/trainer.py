import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import torch_utils
from model.C2DSR import C2DSR
import pdb
import numpy as np

class Trainer(object):
    def __init__(self, opt):
        raise NotImplementedError

    def update(self, batch):
        raise NotImplementedError

    def predict(self, batch):
        raise NotImplementedError

    def update_lr(self, new_lr):  # here should change
        torch_utils.change_lr(self.optimizer, new_lr)

    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.load_state_dict(checkpoint['model'])
        self.opt = checkpoint['config']

    def save(self, filename, epoch):
        params = {
            'model': self.model.state_dict(),
            'config': self.opt,
        }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

class CDSRTrainer(Trainer):
    def __init__(self, opt, adj = None, adj_single = None):
        self.opt = opt
        if opt["model"] == "C2DSR":
            self.model = C2DSR(opt, adj, adj_single)
        else:
            print("please select a valid model")
            exit(0)

        self.mi_loss = 0
        self.BCE_criterion = nn.BCEWithLogitsLoss()
        self.CS_criterion = nn.CrossEntropyLoss(reduction='none')
        if opt['cuda']:
            self.model.cuda()
            self.BCE_criterion.cuda()
            self.CS_criterion.cuda()
        self.optimizer = torch_utils.get_optimizer(opt['optim'], self.model.parameters(), opt['lr'])

    def get_dot_score(self, A_embedding, B_embedding):
        output = (A_embedding * B_embedding).sum(dim=-1)
        return output

    def unpack_batch_predict(self, batch):
        if self.opt["cuda"]:
            inputs = [Variable(b.cuda()) for b in batch]
            seq = inputs[0]
            x_seq = inputs[1]
            y_seq = inputs[2]
            position = inputs[3]
            x_position = inputs[4]
            y_position = inputs[5]
            X_last = inputs[6]
            Y_last = inputs[7]
            XorY = inputs[8]
            ground_truth = inputs[9]
            neg_list = inputs[10]
        else:
            inputs = [Variable(b) for b in batch]
            seq = inputs[0]
            x_seq = inputs[1]
            y_seq = inputs[2]
            position = inputs[3]
            x_position = inputs[4]
            y_position = inputs[5]
            X_last = inputs[6]
            Y_last = inputs[7]
            XorY = inputs[8]
            ground_truth = inputs[9]
            neg_list = inputs[10]
        return seq, x_seq, y_seq, position, x_position, y_position, X_last, Y_last, XorY, ground_truth, neg_list

    def unpack_batch(self, batch):
        if self.opt["cuda"]:
            inputs = [Variable(b.cuda()) for b in batch]
            seq = inputs[0]
            x_seq = inputs[1]
            y_seq = inputs[2]
            position = inputs[3]
            x_position = inputs[4]
            y_position = inputs[5]
            ground = inputs[6]
            share_x_ground = inputs[7]
            share_y_ground = inputs[8]
            x_ground = inputs[9]
            y_ground = inputs[10]
            ground_mask = inputs[11]
            share_x_ground_mask = inputs[12]
            share_y_ground_mask = inputs[13]
            x_ground_mask = inputs[14]
            y_ground_mask = inputs[15]
            corru_x = inputs[16]
            corru_y = inputs[17]
        else:
            inputs = [Variable(b) for b in batch]
            seq = inputs[0]
            x_seq = inputs[1]
            y_seq = inputs[2]
            position = inputs[3]
            x_position = inputs[4]
            y_position = inputs[5]
            ground = inputs[6]
            share_x_ground = inputs[7]
            share_y_ground = inputs[8]
            x_ground = inputs[9]
            y_ground = inputs[10]
            ground_mask = inputs[11]
            share_x_ground_mask = inputs[12]
            share_y_ground_mask = inputs[13]
            x_ground_mask = inputs[14]
            y_ground_mask = inputs[15]
            corru_x = inputs[16]
            corru_y = inputs[17]
        return seq, x_seq, y_seq, position, x_position, y_position, ground, share_x_ground, share_y_ground, x_ground, y_ground, ground_mask, share_x_ground_mask, share_y_ground_mask, x_ground_mask, y_ground_mask, corru_x, corru_y

    def HingeLoss(self, pos, neg):
        gamma = torch.tensor(self.opt["margin"])
        if self.opt["cuda"]:
            gamma = gamma.cuda()
        return F.relu(gamma - pos + neg).mean()


    def train_batch(self, batch):
        self.model.train()
        self.optimizer.zero_grad()
        self.model.graph_convolution()

        seq, x_seq, y_seq, position, x_position, y_position, ground, share_x_ground, share_y_ground, x_ground, y_ground, ground_mask, share_x_ground_mask, share_y_ground_mask, x_ground_mask, y_ground_mask, corru_x, corru_y = self.unpack_batch(batch)
        seqs_fea, x_seqs_fea, y_seqs_fea = self.model(seq, x_seq, y_seq, position, x_position, y_position)

        corru_x_fea = self.model.false_forward(corru_x, position)
        corru_y_fea = self.model.false_forward(corru_y, position)

        x_mask = x_ground_mask.float().sum(-1).unsqueeze(-1).repeat(1,x_ground_mask.size(-1))
        x_mask = 1 / x_mask
        x_mask = x_ground_mask * x_mask # for mean
        x_mask = x_mask.unsqueeze(-1).repeat(1,1,seqs_fea.size(-1))
        r_x_fea = (x_seqs_fea * x_mask).sum(1)

        y_mask = y_ground_mask.float().sum(-1).unsqueeze(-1).repeat(1, y_ground_mask.size(-1))
        y_mask = 1 / y_mask
        y_mask = y_ground_mask * y_mask # for mean
        y_mask = y_mask.unsqueeze(-1).repeat(1,1,seqs_fea.size(-1))
        r_y_fea = (y_seqs_fea * y_mask).sum(1)


        real_x_fea = (seqs_fea * x_mask).sum(1)
        real_y_fea = (seqs_fea * y_mask).sum(1)
        x_false_fea = (corru_x_fea * x_mask).sum(1)
        y_false_fea = (corru_y_fea * y_mask).sum(1)


        real_x_score = self.model.D_X(r_x_fea, real_y_fea) # the cross-domain infomax
        false_x_score = self.model.D_X(r_x_fea, y_false_fea)

        real_y_score = self.model.D_Y(r_y_fea, real_x_fea)
        false_y_score = self.model.D_Y(r_y_fea, x_false_fea)

        pos_label = torch.ones_like(real_x_score).cuda()
        neg_label = torch.zeros_like(false_x_score).cuda()
        x_mi_real = self.BCE_criterion(real_x_score, pos_label)
        x_mi_false = self.BCE_criterion(false_x_score, neg_label)
        x_mi_loss = x_mi_real + x_mi_false

        y_mi_real = self.BCE_criterion(real_y_score, pos_label)
        y_mi_false = self.BCE_criterion(false_y_score, neg_label)
        y_mi_loss = y_mi_real + y_mi_false

        used = 10
        ground = ground[:,-used:]
        ground_mask = ground_mask[:, -used:]
        share_x_ground = share_x_ground[:, -used:]
        share_x_ground_mask = share_x_ground_mask[:, -used:]
        share_y_ground = share_y_ground[:, -used:]
        share_y_ground_mask = share_y_ground_mask[:, -used:]
        x_ground = x_ground[:, -used:]
        x_ground_mask = x_ground_mask[:, -used:]
        y_ground = y_ground[:, -used:]
        y_ground_mask = y_ground_mask[:, -used:]


        share_x_result =  self.model.lin_X(seqs_fea[:,-used:]) # b * seq * X_num
        share_y_result = self.model.lin_Y(seqs_fea[:, -used:])  # b * seq * Y_num
        share_pad_result = self.model.lin_PAD(seqs_fea[:, -used:])  # b * seq * 1
        share_trans_x_result = torch.cat((share_x_result, share_pad_result), dim=-1)
        share_trans_y_result = torch.cat((share_y_result, share_pad_result), dim=-1)


        specific_x_result = self.model.lin_X(seqs_fea[:,-used:] + x_seqs_fea[:, -used:])  # b * seq * X_num
        specific_x_pad_result = self.model.lin_PAD(x_seqs_fea[:, -used:])  # b * seq * 1
        specific_x_result = torch.cat((specific_x_result, specific_x_pad_result), dim=-1)

        specific_y_result = self.model.lin_Y(seqs_fea[:,-used:] + y_seqs_fea[:, -used:])  # b * seq * Y_num
        specific_y_pad_result = self.model.lin_PAD(y_seqs_fea[:, -used:])  # b * seq * 1
        specific_y_result = torch.cat((specific_y_result, specific_y_pad_result), dim=-1)

        x_share_loss = self.CS_criterion(
            share_trans_x_result.reshape(-1, self.opt["source_item_num"] + 1),
            share_x_ground.reshape(-1))  # b * seq
        y_share_loss = self.CS_criterion(
            share_trans_y_result.reshape(-1, self.opt["target_item_num"] + 1),
            share_y_ground.reshape(-1))  # b * seq
        x_loss = self.CS_criterion(
            specific_x_result.reshape(-1, self.opt["source_item_num"] + 1),
            x_ground.reshape(-1))  # b * seq
        y_loss = self.CS_criterion(
            specific_y_result.reshape(-1, self.opt["target_item_num"] + 1),
            y_ground.reshape(-1))  # b * seq

        x_share_loss = (x_share_loss * (share_x_ground_mask.reshape(-1))).mean()
        y_share_loss = (y_share_loss * (share_y_ground_mask.reshape(-1))).mean()
        x_loss = (x_loss * (x_ground_mask.reshape(-1))).mean()
        y_loss = (y_loss * (y_ground_mask.reshape(-1))).mean()

        loss = self.opt["lambda"]*(x_share_loss + y_share_loss + x_loss + y_loss) + (1 - self.opt["lambda"]) * (x_mi_loss + y_mi_loss)

        self.mi_loss += (1 - self.opt["lambda"]) * (x_mi_loss.item() + y_mi_loss.item())
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def test_batch(self, batch):
        seq, x_seq, y_seq, position, x_position, y_position, X_last, Y_last, XorY, ground_truth, neg_list = self.unpack_batch_predict(batch)
        seqs_fea, x_seqs_fea, y_seqs_fea = self.model(seq, x_seq, y_seq, position, x_position, y_position)

        X_pred = []
        Y_pred = []
        for id, fea in enumerate(seqs_fea): # b * s * f
            if XorY[id] == 0:
                share_fea = seqs_fea[id, -1]
                specific_fea = x_seqs_fea[id, X_last[id]]
                X_score = self.model.lin_X(share_fea + specific_fea).squeeze(0)
                cur = X_score[ground_truth[id]]
                score_larger = (X_score[neg_list[id]] > (cur + 0.00001)).data.cpu().numpy()
                true_item_rank = np.sum(score_larger) + 1
                X_pred.append(true_item_rank)

            else :
                share_fea = seqs_fea[id, -1]
                specific_fea = y_seqs_fea[id, Y_last[id]]
                Y_score = self.model.lin_Y(share_fea + specific_fea).squeeze(0)
                cur = Y_score[ground_truth[id]]
                score_larger = (Y_score[neg_list[id]] > (cur + 0.00001)).data.cpu().numpy()
                true_item_rank = np.sum(score_larger) + 1
                Y_pred.append(true_item_rank)

        return X_pred, Y_pred