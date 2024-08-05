from torchvision import transforms
import torch
import PIL
from feature_aggregator import FeatureAggregator
from cutpaste import CutPaste
from utils import load_backbone
import numpy as np
import torch.nn.functional as F

# features sextractor
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
class featuresExtractor():
    def __init__(self, normal_files,args,device):
        self.resize = args.resize_size[0]
        self.imagesize = args.target_size[0]
        self.transform_img = transforms.Compose([
            transforms.Resize(self.resize),
            transforms.CenterCrop(self.imagesize),
            transforms.ToTensor(),
        ])
        self.transform_mask = transforms.Compose([
            transforms.Resize(self.resize,PIL.Image.NEAREST),
            transforms.CenterCrop(self.imagesize),
            transforms.ToTensor(),
        ])
        self.imagesize = (3,self.imagesize,self.imagesize)
        self.cutpaste = CutPaste(normal_files,args.class_name)
        backbone = load_backbone(args.backbone, args.edc)
        self.featureAggregator = FeatureAggregator(
            backbone = backbone,
            layers = args.layers,
            target_size = args.target_size,
            patch_size = args.patch_size,
            stride=1,
            output_dim = args.preprocessing_dimension,
            target_dim = args.target_embed_dimension,
            device = device
        ).to(device)
        self.device = device
    def get_features(self, image_path, mask_path):
        images = []
        masks = []
        B = len(image_path)
        for img in image_path:
            image = PIL.Image.open(img).convert('RGB')
            image = self.transform_img(image)
            image = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)(image).float()
            images.append(image)
        images = torch.stack(images,dim=0).to(self.device)
        data, patch_shapes = self.featureAggregator(images)

        h,w = patch_shapes[0]
        data = data.reshape(B,h,w,-1).permute(0,3,1,2)
        data = F.interpolate(data,size=(self.imagesize[1],self.imagesize[2]),mode='bilinear',align_corners=False)
        data = data.permute(0,2,3,1)
        data = data.detach().cpu().numpy()
        if mask_path:
            for m in mask_path:
                mask = PIL.Image.open(m)
                mask = self.transform_mask(mask).squeeze().cpu().numpy()
                masks.append(mask)
            masks = np.array(masks)
            masks = masks>0
            return images,data, masks
        else:
            return images,data, None
    def get_anomalous_features(self, image_path, mask_path):
        images = []
        masks = []
        B = len(image_path)

        for img,m in zip(image_path,mask_path):
            #image = PIL.Image.open(img).convert('RGB')
            image, mask = self.cutpaste.copy_paste(img,m)
            image = PIL.Image.fromarray(image)
            mask = PIL.Image.fromarray(mask)
            image = self.transform_img(image)
            image = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)(image).float()
            #mask = PIL.Image.open(m)
            mask = self.transform_mask(mask).squeeze().cpu().numpy()
            
            images.append(image)
            masks.append(mask)
        images = torch.stack(images,dim=0).to(self.device)
        masks = np.array(masks)
        
        data, patch_shapes = self.featureAggregator(images)
        h,w = patch_shapes[0]
        data = data.reshape(B,h,w,-1).permute(0,3,1,2)
        data = F.interpolate(data,size=(self.imagesize[1],self.imagesize[2]),mode='bilinear',align_corners=False)
        data = data.permute(0,2,3,1)
        data = data.detach().cpu().numpy()
         
        return images,data, masks
