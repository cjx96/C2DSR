import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.nn as nn
from torch.nn.modules.module import Module
import numpy as np



class GCNLayer(nn.Module):
    """
        GCN Module layer
    """
    def __init__(self, opt):
        super(GCNLayer, self).__init__()
        self.opt=opt
        self.dropout = opt["dropout"]
        self.layer_number = opt["GNN"]

        self.encoder = []
        for i in range(self.layer_number):
            self.encoder.append(GNN(
                nfeat=opt["hidden_units"],
                nhid=opt["hidden_units"],
                dropout=opt["dropout"],
                alpha=opt["leakey"]))

        self.encoder = nn.ModuleList(self.encoder)

    def forward(self, fea, adj):
        learn_fea = fea
        tmp_fea = fea
        for layer in self.encoder:
            learn_fea = F.dropout(learn_fea, self.dropout, training=self.training)
            learn_fea = layer(learn_fea, adj)
            tmp_fea = tmp_fea + learn_fea
        return tmp_fea / (self.layer_number + 1)


class GNN(nn.Module):
    def __init__(self, nfeat, nhid, dropout, alpha):
        super(GNN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.dropout = dropout
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, x, adj):
        x = self.gc1(x, adj)
        return x


class GraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        # self.weight = self.glorot_init(in_features, out_features)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def glorot_init(self, input_dim, output_dim):
        init_range = np.sqrt(6.0 / (input_dim + output_dim))
        initial = torch.rand(input_dim, output_dim) * 2 * init_range - init_range
        return nn.Parameter(initial / 2)

    def forward(self, input, adj):
        support = input
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
