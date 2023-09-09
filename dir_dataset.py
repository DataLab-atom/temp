import torch
from torch_geometric.data import InMemoryDataset

class DIRDATASET(InMemoryDataset):
    #splits = ['test', 'train']
    def __init__(self,name,b, mode = 'train', root = 'data/dir_data/', transform=None, pre_transform=None):
        # 数据的下载和处理过程在父类中调用实现
        super(DIRDATASET, self).__init__(root, transform, pre_transform)
        self.name = name
        # 加载数据
        self.data, self.slices = torch.load(root + name + '/75sp_{}_'.format(b) + mode + '.pt')
        self.data.y = self.data.y.long() 
        print(self.data)

    def __repr__(self):
        return '{}({})'.format(self.name, len(self))
