import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, image_path, label_path):
        super(MyDataset, self).__init__()
        self.image_path = glob.glob(os.path.join(image_path, '*.npy'))
        self.label_path = glob.glob(os.path.join(label_path, '*.npy'))
        
        
    def __len__(self):
        return len(self.image_path)
    
    def z_score_normalize(self, data):
        """ Apply Z-score normalization: (data - mean) / std """
        mean = np.mean(data)
        std = np.std(data)
        return (data - mean) / std if std > 0 else np.zeros_like(data)

    def __getitem__(self, index):
        image_data = np.load(self.image_path[index])
        label_data = np.load(self.label_path[index])
        
        # image_data = self.z_score_normalize(image_data)
        # label_data = self.z_score_normalize(label_data)
        
        image_data = torch.from_numpy(image_data)
        label_data = torch.from_numpy(label_data)
        
        image_data.unsqueeze_(0)
        label_data.unsqueeze_(0)
        
        return image_data, label_data

if __name__ == "__main__":

    image_path = './image/'
    label_path = './label/'
    seismic_dataset = MyDataset(image_path, label_path)
    
    train_loader = torch.utils.data.DataLoader(dataset=seismic_dataset,
                                               batch_size=4,
                                               shuffle=True)
    print('Dataset size:', len(seismic_dataset))
    print('train_loader:', len(train_loader))
