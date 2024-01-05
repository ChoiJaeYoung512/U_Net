import os
import numpy as np
import matplotlib.pyplot as plt
import dataloader_v1
import torch
import model as model
import torchvision
import tqdm
import copy
import sys
from torchmetrics import AveragePrecision
from torch.utils.data import DataLoader
from torchvision import transforms
from monai.losses.dice import DiceLoss
# -------------------------------------------
from load_save import *
from util import *

from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# /** 안쓰는 함수 */


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

transform_img = transforms.Compose([transforms.Resize((512,512))
                                    , transforms.ToTensor()
                                    , transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
                                
transform_seg = transforms.Compose([transforms.Resize((512,512))
                                    , transforms.ToTensor()])

dataset_name = "CrackLs315"                                               ################################### 실행시 확인 필! #######################################

dataset_test = dataloader_v1.Dataset(data_dir='/workspace/Code/U-Net_v1/crack_dataset/'+ dataset_name +'/Test/',transform_img=transform_img, transform_seg=transform_seg)
loader_test = DataLoader(dataset_test, batch_size=8, shuffle=False)

# 그밖에 부수적인 variables 설정하기
num_data_test = len(dataset_test)
num_batch_test = np.ceil(num_data_test / 8) # 나눈 값은 batch size

ckpt_dir = './ckpt_' + dataset_name

# 결과 디렉토리 생성하기
result_dir = os.path.join('./', dataset_name + '_result')
if not os.path.exists(result_dir):
    os.makedirs(os.path.join(result_dir, 'png', 'Label'))
    os.makedirs(os.path.join(result_dir, 'png', 'Input'))
    os.makedirs(os.path.join(result_dir, 'png', 'Output'))
    os.makedirs(os.path.join(result_dir, 'numpy', 'Label'))
    os.makedirs(os.path.join(result_dir, 'numpy', 'Input'))
    os.makedirs(os.path.join(result_dir, 'numpy', 'Output'))
    


net = model.UNet()
net.to(device)

optim = torch.optim.Adam(net.parameters(), lr=0.0001)

fn_loss = torch.nn.BCEWithLogitsLoss().to(device)

dice = DiceLoss(reduction='mean')

net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)


with torch.no_grad():
      net.eval()
      loss_arr = []
      counter = 0
      total_dice_arr = 0.0
      for batch, data in enumerate(loader_test, 1):
          dice_arr = []
          
          ap = []
          
          # forward pass
          label = data['label'].to(device)
          input = data['input'].to(device)
          
          output = net(input)
          # print(output)
          
          ''' check
          # print("code debuging : ", input[1].squeeze().to('cpu').permute(1,2,0))
          # print("code debuging : ", input[1].squeeze().to('cpu'))
          # print("code debuging : ", label[0].squeeze().to('cpu'))
          # sys.exit()
          '''
          
          output = scale_output(output)

          TH = output[output > output.mean()].mean()
          output[output > output.mean()] = 1
          output[output <= output.mean()] = 0
          
          print('#####################################################################################') 
          print(label.size())
      #     print(label[0][0][0].size())    
          print('#####################################################################################')
          print(output.size())
          
          # 손실함수 계산하기
          loss = fn_loss(output, label)
          loss_arr += [loss.item()]
          
          diceLoss = dice(output, label)
          dice_arr += [diceLoss.item()]
          
      #     average_precision = AveragePrecision(task="binary")
      #     ap = average_precision(output, label)
      #     print(ap)
          
          
          
          # iou = iou_pytorch(outputs=output, labels=label)
          
          print("TEST: BATCH %04d / %04d | LOSS %.4f | DICE_LOSS %.4f" %
                (batch, num_batch_test, np.mean(loss_arr), np.mean(dice_arr)))
          total_dice_arr += np.mean(dice_arr)
          counter+=1
          print(total_dice_arr, counter)
          # 테스트 결과 저장하기
          
          for j in range(label.shape[0]):
              id = num_batch_test * (batch - 1) + j

              plt.imsave(os.path.join(result_dir, 'png', 'Label', 'label_%04d.png' % id), label[j].squeeze().to('cpu'), cmap='gray')
            #   print(input[j].permute(1,2,0))
              torchvision.utils.save_image(input[j], os.path.join(result_dir, 'png', 'Input', 'input_%04d.png' % id))
            #   plt.imsave(os.path.join(result_dir, 'png', 'Input', 'input_%04d.png' % id), input[j].squeeze().permute(1,2,0).to('cpu').numpy(), cmap='gray')
              plt.imsave(os.path.join(result_dir, 'png', 'Output', 'output_%04d.png' % id), output[j].squeeze().to('cpu'), cmap='gray')

              np.save(os.path.join(result_dir, 'numpy', 'Label', 'label_%04d.npy' % id), label[j].squeeze().to('cpu'))
              np.save(os.path.join(result_dir, 'numpy', 'Input', 'input_%04d.npy' % id), input[j].squeeze().to('cpu'))
              np.save(os.path.join(result_dir, 'numpy', 'Output', 'output_%04d.npy' % id), output[j].squeeze().to('cpu'))
              
print("AVERAGE TEST: BATCH %04d / %04d | LOSS %.4f | DICE_LOSS %.4f" %
      (batch, num_batch_test, np.mean(loss_arr), np.mean(total_dice_arr)/counter))
sys.exit()



lst_data = os.listdir(os.path.join(result_dir, 'numpy'))
lst_label = [f for f in lst_data if f.startswith('label')]
lst_input = [f for f in lst_data if f.startswith('input')]
lst_output = [f for f in lst_data if f.startswith('output')]

lst_label.sort()
lst_input.sort()
lst_output.sort()

##
id = 0

label = np.load(os.path.join(result_dir,"numpy", lst_label[id]))
input = np.load(os.path.join(result_dir,"numpy", lst_input[id]))
output = np.load(os.path.join(result_dir,"numpy", lst_output[id]))

## 플롯 그리기
plt.figure(figsize=(8,6))
plt.subplot(131)
plt.imshow(input, cmap='gray')
plt.title('input')

plt.subplot(132)
plt.imshow(label, cmap='gray')
plt.title('label')

plt.subplot(133)
plt.imshow(output, cmap='gray')
plt.title('output')

plt.show()