import torch
import random
import numpy as  np
import torchvision
import scipy
import pickle
import os
from skimage.segmentation import slic
color = [i for i in range(0,255,28)]

#img_data to img_data_sp
def process_image(img, index, n_images, to_print, shuffle,n_sp = 75,dataset = 'mnist',compactness = 0.25,split = 'train'):
    assert img.dtype == np.uint8, img.dtype
    img = (img / 255.).astype(np.float32)

    n_sp_extracted = n_sp + 1  # number of actually extracted superpixels (can be different from requested in SLIC)
    n_sp_query = n_sp + (20 if dataset == 'mnist' else 50)  # number of superpixels we ask to extract (larger to extract more superpixels - closer to the desired n_sp)
    while n_sp_extracted > n_sp:
        superpixels = slic(img, n_segments=n_sp_query, compactness=compactness, multichannel=len(img.shape) > 2)
        sp_indices = np.unique(superpixels)
        n_sp_extracted = len(sp_indices)
        n_sp_query -= 1  # reducing the number of superpixels until we get <= n superpixels

    assert n_sp_extracted <= n_sp and n_sp_extracted > 0, (split, index, n_sp_extracted, n_sp)
    assert n_sp_extracted == np.max(superpixels), ('superpixel indices', np.unique(superpixels))  # make sure superpixel indices are numbers from 0 to n-1

    if shuffle:
        ind = np.random.permutation(n_sp_extracted)
    else:
        ind = np.arange(n_sp_extracted)

    sp_order = sp_indices[ind].astype(np.int32)
    if len(img.shape) == 2:
        img = img[:, :, None]

    n_ch = 1 if img.shape[2] == 1 else 3

    sp_intensity, sp_coord = [], []
    for seg in sp_order:
        mask = (superpixels == seg).squeeze()
        avg_value = np.zeros(n_ch)
        for c in range(n_ch):
            avg_value[c] = np.mean(img[:, :, c][mask])
        cntr = np.array(scipy.ndimage.measurements.center_of_mass(mask))  # row, col
        sp_intensity.append(avg_value)
        sp_coord.append(cntr)
    sp_intensity = np.array(sp_intensity, np.float32)
    sp_coord = np.array(sp_coord, np.float32)
    if to_print:
        print('image={}/{}, shape={}, min={:.2f}, max={:.2f}, n_sp={}'.format(index + 1, n_images, img.shape,
                                                                              img.min(), img.max(), sp_intensity.shape[0]))

    return sp_intensity, sp_coord, sp_order, superpixels


def get_bais_data(data,labels,bias):
    index =  torch.randperm(len(data))
    bais_data_index = index[:int(bias*len(data))]
    no_bais_data_index = index[int(bias*len(data)):]

    a = torch.ones([28,28])
    for i in no_bais_data_index:
        b = a*(random.randint(0,255))
        data[i] = torch.where(data[i] == 0, b , data[i])

    for i in bais_data_index:
        b = a*color[labels[i]]
        data[i] = torch.where(data[i] == 0, b , data[i])
    
    return data,bais_data_index,no_bais_data_index

def save_bias_data(data,lable,sp,dataset_name,bias,split = 'tarin'):
    data_dir = 'data/bias_data/{}'.format(dataset_name)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    with open(data_dir + '/{}sp_{}_{}.pkl'.format(sp,bias,split), 'wb') as f:
        pickle.dump((np.array(data).astype(np.int32), lable), f, protocol=2)

def save_sp_bias_data(lable,sp_data,sp,dataset_name,bias,split = 'tarin'):
    data_dir = 'data/sp_bias_data/{}'.format(dataset_name)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    sp_data = [sp_data[i][:3] for i in range(len(lable))]
    with open(data_dir + '/{}sp_{}_{}.pkl'.format(sp,bias,split), 'wb') as f:
        pickle.dump((np.array(lable).astype(np.int32), sp_data), f, protocol=2)

#data_setname in ['mnist','kuzu','fashion']
def get_data(data_setname,split = 'train'):
    if data_setname == 'mnist':
        Data_set = torchvision.datasets.MNIST('data/data/MNIST/',train=(split == 'train'),download=True,transform = torchvision.transforms.ToTensor())
    elif data_setname == 'fashion':
        Data_set = torchvision.datasets.FashionMNIST('data/data/FashionMNIST/',train=(split == 'train'),download=True,transform = torchvision.transforms.ToTensor())
    elif data_setname == 'kuzu':
        Data_set = torchvision.datasets.KMNIST('data/data/KMNIST/',train=(split == 'train'),download=True,transform = torchvision.transforms.ToTensor())

    return Data_set.data,Data_set.targets


