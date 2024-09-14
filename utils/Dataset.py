from torch.utils.data import Dataset
import pdb


class HybridDataset(Dataset):
    def __init__(self, src, dst):
        super(HybridDataset).__init__()
        self.src = src
        self.dst = dst
    
    def __len__(self):
        return len(self.dst)
    
    def __getitem__(self, idx):
        return self.src[idx], self.dst[idx]


class TrainDataset(Dataset):
    def __init__(self, ids, mask, labels):
        super(TrainDataset, self).__init__()
        self.ids = ids
        self.mask = mask
        self.labels = labels
    
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        return self.ids[idx], self.mask[idx], self.labels[idx]


class DynamicDataset(Dataset):
    def __init__(self, *args):
        super(DynamicDataset).__init__()
        self.args = args
    
    def __len__(self):
        return len(self.args[0])
    
    def __getitem__(self, idx):
        return tuple(arg[idx] for arg in self.args)




