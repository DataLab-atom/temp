from typing import Callable, Optional
import torch
import numpy as  np
import pickle
from scipy.spatial.distance import cdist
from torch_geometric.data import Data,InMemoryDataset
from collections import defaultdict
import collections
import os

def normalize(x, eps=1e-7):
    return x / (x.sum() + eps)

def compute_adjacency_matrix_images(coord, sigma=0.1):
    coord = coord.reshape(-1, 2)
    dist = cdist(coord, coord)
    A = np.exp(- dist / (sigma * np.pi) ** 2)
    A[np.diag_indices_from(A)] = 0
    return A


def precompute_graph_images(img_size):
    col, row = np.meshgrid(np.arange(img_size), np.arange(img_size))
    coord = np.stack((col, row), axis=2) / img_size  # 28,28,2
    A = torch.from_numpy(compute_adjacency_matrix_images(coord)).float().unsqueeze(0)
    coord = torch.from_numpy(coord).float().unsqueeze(0).view(1, -1, 2)
    mask = torch.ones(1, img_size * img_size, dtype=torch.uint8)
    return A, coord, mask


def list_to_torch(data):
    for i in range(len(data)):
        if data[i] is None:
            continue
        elif isinstance(data[i], np.ndarray):
            if data[i].dtype == np.bool:
                data[i] = data[i].astype(np.float32)
            data[i] = torch.from_numpy(data[i]).float()
        elif isinstance(data[i], list):
            data[i] = list_to_torch(data[i])
    return data


class MNIST75sp(torch.utils.data.Dataset):
    def __init__(self,
                 data_set_name = 'mnist',
                 sp = 75,
                 bias = 0.9,
                 split = 'train',
                 use_mean_px=True,
                 use_coord=True,
                 gt_attn_threshold=0,
                 attn_coef=None):

        self.split = split
        self.is_test = split.lower() in ['test', 'val']
        with open('data/sp_bias_data/{}/{}sp_{}_{}.pkl'.format(data_set_name,sp,bias,split), 'rb') as f:
            self.labels, self.sp_data = pickle.load(f)

        self.use_mean_px = use_mean_px
        self.use_coord = use_coord
        self.n_samples = len(self.labels)
        self.img_size = 28
        self.gt_attn_threshold = gt_attn_threshold

        self.alpha_WS = None
        if attn_coef is not None and not self.is_test:
            with open(attn_coef, 'rb') as f:
                self.alpha_WS = pickle.load(f)
            print('using weakly-supervised labels from %s (%d samples)' % (attn_coef, len(self.alpha_WS)))

    def train_val_split(self, samples_idx):
        self.sp_data = [self.sp_data[i] for i in samples_idx]
        self.labels = self.labels[samples_idx]
        self.n_samples = len(self.labels)

    def precompute_graph_data(self, replicate_features, threads=0):
        print('precompute all data for the %s set...' % self.split.upper())
        self.Adj_matrices, self.node_features, self.GT_attn, self.WS_attn = [], [], [], []
        for index, sample in enumerate(self.sp_data):
            mean_px, coord = sample[:2]
            coord = coord / self.img_size
            A = compute_adjacency_matrix_images(coord)
            N_nodes = A.shape[0]
            x = None
            if self.use_mean_px:
                x = mean_px.reshape(N_nodes, -1)
            if self.use_coord:
                coord = coord.reshape(N_nodes, 2)
                if self.use_mean_px:
                    x = np.concatenate((x, coord), axis=1)
                else:
                    x = coord
            if x is None:
                x = np.ones(N_nodes, 1)  # dummy features
            if replicate_features:
                x = np.pad(x, ((0, 0), (2, 0)), 'edge')  # replicate features to make it possible to test on colored images
            if self.gt_attn_threshold == 0:
                gt_attn = (mean_px > 0).astype(np.float32)
            else:
                gt_attn = mean_px.copy()
                gt_attn[gt_attn < self.gt_attn_threshold] = 0
            self.GT_attn.append(normalize(gt_attn))

            if self.alpha_WS is not None:
                self.WS_attn.append(normalize(self.alpha_WS[index]))

            self.node_features.append(x)
            self.Adj_matrices.append(A)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        data = [self.node_features[index],
                self.Adj_matrices[index],
                self.Adj_matrices[index].shape[0],
                self.labels[index],
                self.GT_attn[index]]

        if self.alpha_WS is not None:
            data.append(self.WS_attn[index])

        data = list_to_torch(data)  # convert to torch

        return data

    def get_graph_data(self):
        self.precompute_graph_data(1)
        data = {
            'x' : [],
            'edge_index':[],
            'edge_attr':[],
            'edge_attr_sp' : [],
            'y':[],
            'gt_attn':[],
            'index' :[]
        }
        for index in range(len(self.labels)):
            data['index'].append(index)
            x = torch.tensor(self.node_features[index])
            data['x'].append(x)  # 节点特征向量
            A = self.Adj_matrices[index]  # 邻接矩阵
            data['y'].append(self.labels[index])  # 标签
            data['gt_attn'].append(torch.tensor(self.GT_attn[index]).flatten())    # ground truth 注意力图

            edge_attr = []
            edge_attr_sp = []
            edge_index = [[],[]]
            for i in range(x.shape[0]):
                for j in range(0,i):
                    if A[i][j] > 1e-2:
                        edge_index[0].append(i)
                        edge_index[1].append(j)
                        edge_attr_sp.append(A[i][j])
                        edge_attr.append(1)
            data['edge_attr'].append(torch.tensor(edge_attr))  
            data['edge_index'].append(torch.tensor(edge_index))  
            data['edge_attr_sp'].append(torch.tensor(edge_attr_sp))  
        return data


def collate(data):
    length = len(data['x'])
    shapelist = ['x','edge_attr','edge_attr_sp','gt_attn']
    noshapelist = ['y','index']

    slices,temp_slices  =  {},{}
    for i in data.keys():
        slices[i] = []
        temp_slices[i] = 0

    for i in range(length):

        for j in temp_slices.keys():
            slices[j].append(temp_slices[j])

        temp_slices['edge_index'] += data['edge_index'][i].shape[1]
        for j in shapelist:
            temp_slices[j] += data[j][i].shape[0]
        
        for j in noshapelist:
            temp_slices[j] += 1                
    for j in temp_slices.keys():
        slices[j].append(temp_slices[j])
    for i in data.keys():
        slices[i] = torch.tensor(slices[i],dtype = torch.int)
    slice = defaultdict(torch.tensor,slices)
    data = Data(x = torch.cat(data['x']),
                edge_index = torch.cat(data['edge_index'],dim=1),
                edge_attr = torch.cat(data['edge_attr']),
                edge_attr_sp = torch.cat(data['edge_attr_sp']),
                gt_attn = torch.cat(data['gt_attn']),
                y = torch.tensor(data['y']),
                index = torch.tensor(data['index']))

    return data,slice

# data_set_name in ['mnist','kuzu','fashion']
def make_sp_biasdataset(data_set_name,sp = 75,bias = 0.9,split = 'train'):
    data_dir = 'data/GNN_data/{}'.format(data_set_name)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    data_set =  MNIST75sp(data_set_name,sp,bias,split)
    my_data = data_set.get_graph_data()
    data,slices = collate(my_data)

    torch.save((data, slices), data_dir + '/{}sp_{}_{}.pt'.format(sp,bias,split))