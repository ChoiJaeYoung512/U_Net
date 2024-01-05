import dataloader_v1
import torch
import torch.nn as nn
import model as model
import numpy as np
import torchvision
import tqdm
import matplotlib.pyplot as plt
import copy
import sys
import wandb
import torch.optim as Optim
from torch.utils.data import DataLoader
from torchvision import transforms
from monai.losses.dice import DiceLoss

# -------------------------------------------
from load_save import *
from util import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform_img = transforms.Compose([transforms.Resize((512,512))
                                , transforms.ToTensor()
                                , transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
                                
transform_seg = transforms.Compose([transforms.Resize((512,512))
                                , transforms.ToTensor()])

# Convert to jpg format if segment image is in bmp format

data_root = "/workspace/Code/U-Net_v1/crack_dataset/"                           ################################### 실행시 확인 필! #######################################
dataset_name = "Total_data"                                                     ################################### 실행시 확인 필! #######################################

final_path = data_root + dataset_name + '/'

src_path = final_path + "bmp_masks" # jpg images path
dst_path = final_path + "masks/" # bmp images path

if os.path.isfile(src_path):
    format_converter_bmp(src_path=src_path, dst_path=dst_path)

    # Temp line (For dataset CRKWH100)
    format_converter_png(src_path= "/workspace/Code/U-Net_v1/crack_dataset/CRKWH100/png_images", dst_path="/workspace/Code/U-Net_v1/crack_dataset/CRKWH100/images/")

    Train_path, Val_path, Test_path = spliter(data_root=final_path)
    print('############################')
    print('####  convert complete  ####')
    print('############################')
else :
    Train_path = final_path + "Train/"
    Val_path = final_path + "Val/"
    Test_path = final_path + "Test/"
    print('##########################################################################')
    print('####  File does not exist and does not convert: error may occurrence  ####')
    print('##########################################################################')
# sys.exit()

# print(torch.cuda.device_count())


# check point location
ckpt_dir = './ckpt_' + dataset_name

# param
st_epoch = 0
num_epoch = 50
lr = 0.0001
batch_size = 8

# data, dataloader
train_data = dataloader_v1.Dataset(data_dir=Train_path,transform_img=transform_img, transform_seg=transform_seg)
train_dataloader = DataLoader(train_data, batch_size=8, shuffle=True)

val_data = dataloader_v1.Dataset(data_dir=Val_path,transform_img=transform_img, transform_seg=transform_seg)
val_dataloader = DataLoader(val_data, batch_size=8, shuffle=True)

# network
net = model.UNet()
net.to(device)

# data parallel
# model = nn.DataParallel(net)

# optim
optim = torch.optim.Adam(net.parameters(), lr=lr)

# loss
fn_loss = torch.nn.BCEWithLogitsLoss().to(device)
dice = DiceLoss(reduction='mean')

# lr scheduler
scheduler = Optim.lr_scheduler.CosineAnnealingLR(optimizer=optim, T_max=50, eta_min=0)

#wandb
wandb.init(project="wodud")
wandb.run.name = "U_net" + dataset_name
wandb.run.save


args = {
    "learning_rate": lr,
    "epochs": num_epoch,
    "batch_size": batch_size
}

wandb.config.update(args)

for epoch in range(st_epoch + 1, num_epoch + 1):
    net.train()
    loss_item = 0.0
    diceLoss_item = 0.0
    count = 0   

    for batch, data in enumerate(train_dataloader):
        label = data['label'].to(device)
        input = data['input'].to(device)
        
        output = net(input)
        
        #Thresh hold
        # output[output != 1] = 0.
        
        # print(output.dtype)
        # print(label.dtype)
        output = scale_output(output)
        
        # print(output)
        output[output > output.mean()] = 1
        output[output <= output.mean()] = 0
        
        # backward pass
        optim.zero_grad()
        #오차 역전파에 사용하는 계산량을 줄여서 처리 속도를 높임
        
        # diceLoss = DiceLoss(inputs = output, targets = label)
        diceLoss = dice(output, label)
        
        # print("check Line 136 : ", diceLoss.item())
        # sys.exit()
        
        loss = fn_loss(output, label)
        loss.backward()
        optim.step()
        
        diceLoss_item += diceLoss.item()
        loss_item += loss.item()
        count += 1
        # print("check ---------------------------------------")
        # print(loss_item)
        # print(count)
    
    scheduler.step()
    wandb.log({'training loss' : loss_item/count})        
    wandb.log({'training Dice loss' : diceLoss_item/count})     
    wandb.log({'train_images': wandb.Image(input[0]),
               'train_Labels': wandb.Image(label[0]),
               'train_prediction result': wandb.Image(output[0])})
        
        # print(f"TRAIN : EPOCH {epoch :04d} // BATCH {batch : 04d} / {len(train_data) : 04d} / LOSS {np.mean(loss_arr) : .4f}")



    with torch.no_grad():           #gradient계산 context를 비활성화, 필요한 메모리 감소, 연산속도 증가
        net.eval()                  #batch normalization과 dropout등과 같은 학습에만 필요한 기능 비활성화 추론할때의 상태로 조정(메모리와는 연관 없음)
        loss_item = 0.0
        diceLoss_item = 0.0
        count = 0  
        
        best_model_wts = copy.deepcopy(net.state_dict())
        for batch, data in enumerate(val_dataloader, 1):
            # forward pass
            label = data['label'].to(device)
            input = data['input'].to(device)

            output = net(input)

            diceLoss = dice(output, label)
            
            loss = fn_loss(output, label)
            loss_item += loss.item()
            
            diceLoss_item += diceLoss.item()
            count += 1
            
        wandb.log({'validation loss' : loss_item/count})   
        wandb.log({'validation Dice loss' : diceLoss_item/count})     
        wandb.log({'val_images': wandb.Image(input[0]),
                   'val_Labels': wandb.Image(label[0]),
                   'val_prediction result': wandb.Image(output[0])})
            # print(f"VALID : EPOCH {epoch :04d} // BATCH {batch : 04d} / {len(val_data) : 04d} // LOSS {np.mean(loss_arr) : .4f}")
        
    if epoch % 1 == 0:
        save(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch)
        
        
