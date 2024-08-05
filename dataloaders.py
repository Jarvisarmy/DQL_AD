from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import os

from datasets import MVTecDataset, DatasetSplit

def dataloader_MVTec_setup(data_root,
                           class_name,
                           resize_size = (256,256),
                           target_size = (224,224),
                           batch_size=16):
    #data_path = os.path.join(data_root, 'mvtec')
    data_path = data_root
    train_dataset = MVTecDataset(data_path, class_name, resize=resize_size, imagesize=target_size,split=DatasetSplit.TRAIN,
        train_val_split=1.0)
    test_dataset = MVTecDataset(data_path, class_name, resize=resize_size, imagesize=target_size,split=DatasetSplit.TEST)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return {
        'train_loader': train_loader,
        'val_loader': None,
        'test_loader': test_loader
    }

