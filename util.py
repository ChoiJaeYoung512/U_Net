import os
import random
import torch
import shutil
import glob
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

# spliter param(root dir)
# data_root = "/workspace/Code/U-Net_v1/dataset/"

# bmp to jpg 
# src_path = "./jpg_images" # jpg images path
# dst_path = "./bmp_images/" # bmp images path

def average(list):
  return (sum(list) / len(list))

def scale_output(output):
    min_val, max_val = torch.min(output), torch.max(output)
    scaled_output = (output - min_val) / (max_val - min_val)
    
    return scaled_output

def format_converter_bmp(src_path, dst_path):
    if not os.path.isdir(dst_path): # make dst dir if it's not existed
        os.mkdir(dst_path)

    for jpg_path in tqdm(list(set(glob.glob(src_path+"*/*.bmp", recursive=True)))):
        img = Image.open(jpg_path)
        jpg_name = jpg_path.replace("\\", "/").split("/")[-1]
        bmp_name = jpg_name.replace("bmp", "jpg")
        # print(bmp_name)
        img.save(dst_path+bmp_name)
        
        
def format_converter_png(src_path, dst_path):
    if not os.path.isdir(dst_path): # make dst dir if it's not existed
        os.mkdir(dst_path)
        
    for png_path in tqdm(list(set(glob.glob(src_path+"*/*.png", recursive=True)))):
        img = Image.open(png_path)
        png_name = png_path.replace("\\", "/").split("/")[-1]
        jpg_name = png_name.replace("png", "jpg")
        # print(jpg_name)
        img.save(dst_path+jpg_name)

def spliter(data_root):
    dir1_filename = os.listdir(data_root + "images")
    train = random.sample(dir1_filename, round((len(dir1_filename)*7)/10))

    test = [x for x in dir1_filename if x not in train]
    val = random.sample(test, round((len(test)*3)/10))

    test = [x for x in test if x not in val]

    print(len(train)+len(test)+len(val)) # 개수 확인
    
    # 폴더 생성
    def createFolder(directory):
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print ('Error: Creating directory. ' +  directory)

    createFolder(data_root + "Train")
    createFolder(data_root + "Train/" + 'images')
    createFolder(data_root + "Train/" + 'masks')
    createFolder(data_root + "Val")
    createFolder(data_root + "Val/" + 'images')
    createFolder(data_root + "Val/" + 'masks')
    createFolder(data_root + "Test")
    createFolder(data_root + "Test/" + 'images')
    createFolder(data_root + "Test/" + 'masks')

    for i in train:
        shutil.copyfile(os.path.join(data_root, 'images', str(i)), os.path.join(data_root, 'Train', 'images', str(i)))
        shutil.copyfile(os.path.join(data_root, 'masks', str(i)), os.path.join(data_root, 'Train', 'masks', str(i)))

    for i in test:
        shutil.copyfile(os.path.join(data_root, 'images', str(i)), os.path.join(data_root, 'Test', 'images', str(i)))
        shutil.copyfile(os.path.join(data_root, 'masks', str(i)), os.path.join(data_root, 'Test', 'masks', str(i)))

    for i in val:
        shutil.copyfile(os.path.join(data_root, 'images', str(i)), os.path.join(data_root, 'Val', 'images', str(i)))
        shutil.copyfile(os.path.join(data_root, 'masks', str(i)), os.path.join(data_root, 'Val', 'masks', str(i)))
        Train_path = data_root + "Train/"
        Val_path = data_root + "Val/"
        Test_path = data_root + "Test/"
    return Train_path, Val_path, Test_path


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

# class DiceLoss(nn.Module):
#     def __init__(self):
#         super(DiceLoss, self).__init__()

#     def forward(self, inputs, targets, smooth):
        
#         smooth = 1
#         inputs = F.sigmoid(inputs) # sigmoid를 통과한 출력이면 주석처리
        
#         inputs = inputs.view(-1)
#         targets = targets.view(-1)
        
#         intersection = (inputs * targets).sum()                            
#         dice = (2.*intersection + smooth) / (inputs.sum() + targets.sum() + smooth)  
        
#         return 1 - dice 
    

def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor, smooth = 1e-6):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    SMOOTH = smooth
    
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))         # Will be zzero if both are 0
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
    
    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    
    return thresholded.mean()  # Or thresholded.mean() if you are interested in average across the batch

def get_statistics( pred, gt):
    """
    return tp, fp, fn
    """
    tp = np.sum((pred == 1) & (gt == 1))
    fp = np.sum((pred == 1) & (gt == 0))
    fn = np.sum((pred == 0) & (gt == 1))
    return [tp, fp, fn]

def cal_ois_metrics(pred_list, gt_list, thresh_step=0.01):
    final_acc_all = []
    for pred, gt in zip(pred_list, gt_list):
        statistics = []
        for thresh in np.arange(0.0, 1.0, thresh_step):
            gt_img = (gt / 255).astype('uint8')
            pred_img = (pred / 255 > thresh).astype('uint8')
            tp, fp, fn = get_statistics(pred_img, gt_img)
            p_acc = 1.0 if tp == 0 and fp == 0 else tp / (tp + fp)
            r_acc = tp / (tp + fn)

            if p_acc + r_acc == 0:
                f1 = 0
            else:
                f1 = 2 * p_acc * r_acc / (p_acc + r_acc)
            statistics.append([thresh, f1])
        max_f = np.amax(statistics, axis=0)
        final_acc_all.append(max_f[1])
    return np.mean(final_acc_all)


def Avarage_precision(label, output):
    pass