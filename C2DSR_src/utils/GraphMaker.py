import numpy as np
import random
import scipy.sparse as sp
import torch
import codecs
import json
import copy

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


class GraphMaker(object):
    def __init__(self, opt, filename):
        self.opt = opt
        self.user = set()
        self.item = set()
        train_data = []

        def takeSecond(elem):
            return elem[1]
        with codecs.open(filename, "r", encoding="utf-8") as infile:
            for id, line in enumerate(infile):
                res = []
                line = line.strip().split("\t")[2:]
                for w in line:
                    w = w.split("|")
                    res.append((int(w[0]), int(w[1])))
                res.sort(key=takeSecond)
                res_2 = []
                for r in res:
                    res_2.append(r[0])
                train_data.append(res_2)
        
        self.raw_data = train_data
        self.adj, self.adj_single = self.preprocess(train_data, opt)

    def preprocess(self,data,opt):

        VV_edges = []
        VV_edges_single = []

        real_adj = {}

        for seq in data:
            source = -1
            target = -1
            pre = -1
            for d in seq:
                if d not in real_adj:
                    real_adj[d] = set()
                if d < self.opt["source_item_num"]:
                    if source is not -1:
                        if d in real_adj[source]:
                            continue
                        else:
                            VV_edges_single.append([source, d])
                    source = d

                else :
                    if target is not -1:
                        if d in real_adj[target]:
                            continue
                        else:
                            VV_edges_single.append([target, d])
                    target = d

                if pre is not -1:
                    if d in real_adj[pre]:
                        continue
                    VV_edges.append([pre, d])
                pre=d

        VV_edges = np.array(VV_edges)
        VV_edges_single = np.array(VV_edges_single)
        adj = sp.coo_matrix((np.ones(VV_edges.shape[0]), (VV_edges[:, 0], VV_edges[:, 1])),
                               shape=(opt["itemnum"], opt["itemnum"]),
                               dtype=np.float32)
        adj_single = sp.coo_matrix((np.ones(VV_edges_single.shape[0]), (VV_edges_single[:, 0], VV_edges_single[:, 1])),shape=(opt["itemnum"], opt["itemnum"]),dtype=np.float32)

        adj = normalize(adj)
        adj_single = normalize(adj_single)
        adj = sparse_mx_to_torch_sparse_tensor(adj)
        adj_single = sparse_mx_to_torch_sparse_tensor(adj_single)

        print("real graph loaded!")
        return adj, adj_single

