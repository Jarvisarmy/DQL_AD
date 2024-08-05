import numpy as np
import scipy.ndimage as ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import os
from enum import Enum
from torch.utils.data import Dataset
import time
import random

# ------------------------------------------------------------------------------
# for backbone loading
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
   'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth'
}


def load_backbone(backbone='wide_resnet50_2',edc=False):
    if backbone == 'resnet18':
        model = models.resnet18(weights=None)
    elif backbone == 'resnet50':
        model = models.resnet50(weights=None)
    elif backbone == 'wide_resnet50_2':
        model = models.wide_resnet50_2(weights=None)
    
    
    if edc:
        backbone_path = './best_encoder.pth'
        state_dict = torch.load(backbone_path)
    else:
        state_dict_path = model_urls[backbone]
        state_dict = torch.hub.load_state_dict_from_url(state_dict_path,progress=True)
    model.load_state_dict(state_dict)
    return model


# ------------------------------------------------------------------------------
# for getting fiels
def get_files_masks(dataroot,class_name,target_types, unknown_types,train_num, verbose=False):
    trainpath = os.path.join(dataroot,class_name,'train','good')
    testpath = os.path.join(dataroot, class_name,'test')
    maskpath = os.path.join(dataroot, class_name, "ground_truth")
    train_normal_files = sorted(os.listdir(trainpath))
    train_normal_files = [os.path.join(trainpath,x) for x in train_normal_files]

    # test normal files
    test_normal_files = sorted(os.listdir(os.path.join(testpath,'good')))
    test_normal_files = [os.path.join(testpath,'good',x) for x in test_normal_files]
    train_target_files = []
    train_target_masks = []
    test_target_files = []
    test_target_masks = []
    for anomaly in target_types:
        anomaly_path = os.path.join(testpath,anomaly)
        anomaly_files = sorted(os.listdir(anomaly_path))
        for i, file in enumerate(anomaly_files):
            name, extension = os.path.splitext(file)
            mask_file = name+'_mask'+extension
            if i < train_num:
                train_target_files.append(os.path.join(anomaly_path,file))
                train_target_masks.append(os.path.join(maskpath,anomaly,mask_file))
            else:
                test_target_files.append(os.path.join(anomaly_path,file))
                test_target_masks.append(os.path.join(maskpath,anomaly,mask_file))

    # train unknown files
    # test unknown files
    train_unknown_files = []
    train_unknown_masks = []
    test_unknown_files = []
    test_unknown_masks = []
    for anomaly in unknown_types:
        anomaly_path = os.path.join(testpath,anomaly)
        anomaly_files = sorted(os.listdir(anomaly_path))
        for i, file in enumerate(anomaly_files):
            name, extension = os.path.splitext(file)
            mask_file = name+'_mask'+extension
            if i < train_num:
                train_unknown_files.append(os.path.join(anomaly_path,file))
                train_unknown_masks.append(os.path.join(maskpath,anomaly,mask_file))
            else:
                test_unknown_files.append(os.path.join(anomaly_path,file))
                test_unknown_masks.append(os.path.join(maskpath,anomaly,mask_file))   
    
    files_dict = {'train_normal_files': train_normal_files, 
                  'train_target_files': train_target_files,
                  'train_target_masks': train_target_masks,
                  'train_unknown_files': train_unknown_files,
                  'train_unknown_masks': train_unknown_masks,
                  'test_normal_files': test_normal_files,
                  'test_target_files': test_target_files,
                  'test_target_masks': test_target_masks,
                  'test_unknown_files': test_unknown_files,
                  'test_unknown_masks': test_unknown_masks,}
    if verbose:
        for item, value in files_dict.items():
            print("{}: {}".format(item, len(value)))
    return files_dict




class StateType(Enum):
    TARGET = 'target'
    UNKNOWN = 'unknown'
class NumpyDataset(Dataset):
    def __init__(self, numpy_list):
        self.numpy_list = numpy_list
    def __len__(self):
        return len(self.numpy_list)
    def __getitem__(self,idx):
        numpy_array = self.numpy_list[idx]
        tensor = torch.from_numpy(numpy_array)
        return tensor
    

def evaluate(eval_env,agent,n_evals=5):
    scores = 0
    for i in range(n_evals):
        s, done, ret = eval_env.reset(), False,0
        for i in range(100):
            a = agent.act(s, training=False)
            s_prime,r,terminated,truncated, info = eval_env.step(agent.network,a)
            s = s_prime
            ret += r
            done = terminated or truncated
            if done:
              break
        scores += ret
    return np.round(scores/n_evals,4)

def print_time(start_time,end_time):
    execution_time_seconds = end_time - start_time
    hours = int(execution_time_seconds // 3600)
    minutes = int((execution_time_seconds % 3600) // 60)
    seconds = int(execution_time_seconds % 60)
    print("Command Execution Time:", f"{hours} hours, {minutes} minutes, {seconds} seconds")





