

from DQN import DQN
from data import featuresExtractor
import metrics
from utils import get_files_masks
import torch
import numpy as np
import os
from DQN import DQN
import random
import matplotlib.pyplot as plt

def get_scores(checkpoint_path,device):
    agent, args, files_dict = load_model(checkpoint_path)
    target_types = [item for item in os.listdir(os.path.join(args.data_root, args.class_name,'test')) if item != 'good']
    unknown_types = []
    if files_dict is None:
        files_dict = get_files_masks(args.data_root,args.class_name,target_types,unknown_types,1,verbose=False)
    target_imgs = []
    target_scores = []
    target_labels = []
    target_features = []
    unknown_imgs = []
    unknown_scores =[]
    unknown_labels = []
    unknown_features = []
    unknown_abnormal_imgs = []
    unknown_abnormal_scores = []
    unknown_abnormal_labels = []
    unknown_abnormal_features = []
    feature_extractor = featuresExtractor(files_dict['train_normal_files'],args, device)

    for file in files_dict['test_normal_files']:
        images,embeds,_ = feature_extractor.get_features([file],None)
        images = images.squeeze().cpu().numpy()
        embeds = torch.from_numpy(embeds.squeeze()).to(device)
        H,W,dim = embeds.shape
        embeds = embeds.reshape(-1,dim)
        score = agent.network(embeds)
        score_sum = torch.sum(score,dim=1)
        score = score/score_sum.view(-1,1)
        score = score[:,1]
        
        score = score.reshape(H,W).detach().cpu().numpy()
        unknown_imgs.append(images)
        unknown_scores.append(score)
        unknown_labels.append(np.zeros(score.shape,dtype=int))
    for file, mask_file in zip(files_dict['test_unknown_files'],files_dict['test_unknown_masks']):
        image, embed, mask = feature_extractor.get_features([file],[mask_file])
        image = image.squeeze().cpu().numpy()
        mask = mask.squeeze()
        mask = mask > 0
        embed = torch.from_numpy(embed.squeeze()).to(device)
        H,W,dim = embed.shape
        embed = embed.reshape(-1,dim)
        score = agent.network(embed)
        score_sum = torch.sum(score,dim=1)
        score = score/score_sum.view(-1,1)
        score = score[:,1]
        
        score = score.reshape(H,W).detach().cpu().numpy()
        unknown_abnormal_imgs.append(image)
        unknown_abnormal_scores.append(score)
        unknown_abnormal_labels.append(mask)
    for file, mask_file in zip(files_dict['test_target_files'],files_dict['test_target_masks']):
        image, embed, mask = feature_extractor.get_features([file],[mask_file])
        image = image.squeeze().cpu().numpy()
        mask = mask.squeeze()
        mask = mask > 0
        embed = torch.from_numpy(embed.squeeze()).to(device)
        H,W,dim = embed.shape
        embed = embed.reshape(-1,dim)
        score = agent.network(embed)
        score_sum = torch.sum(score,dim=1)
        score = score/score_sum.view(-1,1)
        score = score[:,1]
        
        score = score.reshape(H,W).detach().cpu().numpy()
        target_imgs.append(image)
        target_scores.append(score)
        target_labels.append(mask)

    unknown_imgs = np.array(unknown_imgs)
    unknown_scores = np.array(unknown_scores)
    unknown_labels = np.array(unknown_labels)
    unknown_abnormal_imgs = np.array(unknown_abnormal_imgs)
    unknown_abnormal_scores = np.array(unknown_abnormal_scores)
    unknown_abnormal_labels = np.array(unknown_abnormal_labels)
    target_imgs = np.array(target_imgs)
    target_scores = np.array(target_scores)
    target_labels = np.array(target_labels)
    total_scores = np.concatenate((target_scores,unknown_scores),axis=0)
    total_labels = np.concatenate((target_labels,unknown_labels),axis=0)
    total_imgs = np.concatenate((target_imgs,unknown_imgs),axis=0)
    files = files_dict['test_target_files'] + files_dict['test_normal_files']
    scores_dict = {
        "files": files,
        "total_scores": total_scores,
        "total_labels": total_labels,
        "total_imgs": total_imgs
    }
    return scores_dict

def evaluate_metrics(target_labels, target_scores_norm):
    auroc = metrics.auroc_score(target_labels,target_scores_norm)
    auprc = metrics.auprc_score(target_labels,target_scores_norm)

    pro = metrics.pro_score(target_labels,target_scores_norm)
    thres = metrics.return_best_thr(target_labels,target_scores_norm)
    target_labels_ravel = target_labels.ravel()
    target_scores_norm_ravel = target_scores_norm.ravel()
    image_labels = np.amax(target_labels,axis=(1,2))
    image_scores = np.amax(target_scores_norm,axis=(1,2))
    image_min = image_scores.min()
    image_max = image_scores.max()
    if image_min != image_max:
        image_scores_norm = (image_scores-image_min)/(image_max-image_min)
    else:
        image_scores_norm = image_scores
    #image_scores_norm = image_scores
    image_thres = metrics.return_best_thr(image_labels,image_scores_norm)
    i_auroc = metrics.auroc_score(image_labels,image_scores_norm)
    i_auprc = metrics.auprc_score(image_labels,image_scores_norm)
    f1 = metrics.f1_score(image_labels, image_scores_norm>=image_thres)
    accuracy = metrics.accuracy_score(image_labels, image_scores_norm>=image_thres)
    result = {
        'auroc': auroc,
        'auprc': auprc,
        'aupro': pro,
        'thres': thres,
        'image_thres':image_thres,
        'i_auroc': i_auroc,
        'i_auprc': i_auprc,
        'f1': f1,
        'accuracy': accuracy
    }
    for item,value in result.items():
        print('{}: {}'.format(item,value))
    return result

def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    args = checkpoint['args']
    state_dict = checkpoint['model']
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
    agent = DQN(args,device=device)
    agent.network.load_state_dict(state_dict)
    files_dict = checkpoint['files_dict'] if 'files_dict' in checkpoint else None
    return agent, args, files_dict                               

def visualize_result(n,target_imgs,target_labels,target_scores_norm):
    idxs = random.sample(list(range(target_imgs.shape[0])),n)
    fig, axs = plt.subplots(3,n,figsize=(n,3))
    for i in range(n):
        idx = idxs[i]
        img_min = target_imgs[idx].min()
        img_max = target_imgs[idx].max()

        img = (target_imgs[idx]-img_min)/(img_max-img_min)
        
        axs[0,i].imshow(img.transpose(1,2,0))
        axs[0,i].set_title('image')
        axs[0,i].axis('off')
        axs[1,i].imshow(target_labels[idx],cmap='gray')
        axs[1,i].set_title('mask')
        axs[1,i].axis('off')
        axs[2,i].imshow(target_scores_norm[idx],cmap='gray')
        axs[2,i].set_title('anomaly map')
        axs[2,i].axis('off')
        print(target_scores_norm[idx].max())
    plt.tight_layout()
    plt.show()
