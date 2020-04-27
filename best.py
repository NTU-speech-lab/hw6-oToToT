#!/usr/bin/env python
# coding: utf-8

# In[1]:


# # 下載資料
# !gdown --id '14CqX3OfY9aUbhGp4OpdSHLvq2321fUB7' --output data.zip
# # 解壓縮
# !unzip data.zip


# In[2]:

import sys
import os
# 讀取 label.csv
import pandas as pd
# 讀取圖片
from PIL import Image
import numpy as np

import torch
# Loss function
import torch.nn.functional as F
# 讀取資料
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
# 載入預訓練的模型
import torchvision.models as models
# 將資料轉換成符合預訓練模型的形式
import torchvision.transforms as transforms
# 顯示圖片
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using", "cuda" if torch.cuda.is_available() else "cpu")


# In[3]:


# 實作一個繼承 torch.utils.data.Dataset 的 Class 來讀取圖片
class Adverdataset(Dataset):
    def __init__(self, root, label, transforms):
        # 圖片所在的資料夾
        self.root = root
        # 由 main function 傳入的 label
        self.label = torch.from_numpy(label).long()
        # 由 Attacker 傳入的 transforms 將輸入的圖片轉換成符合預訓練模型的形式
        self.transforms = transforms
        # 圖片檔案名稱的 list
        self.fnames = []

        for i in range(200):
            self.fnames.append("{:03d}".format(i))

    def __getitem__(self, idx):
        # 利用路徑讀取圖片
        img = Image.open(os.path.join(self.root, self.fnames[idx] + '.png'))
        # 將輸入的圖片轉換成符合預訓練模型的形式
        img = self.transforms(img)
        # 圖片相對應的 label
        label = self.label[idx]
        return img, label
    
    def __len__(self):
        # 由於已知這次的資料總共有 200 張圖片 所以回傳 200
        return 200


# In[4]:


class Attacker:
    def __init__(self, img_dir, label):
        # 讀入預訓練模型 vgg16
        self.model = models.densenet121(pretrained=True)
        self.model.to(DEVICE)
        self.model.eval()
        self.test_models = [
#             models.alexnet(pretrained=True),
#             models.vgg11(pretrained=True),
#             models.vgg11_bn(pretrained=True),
#             models.vgg13(pretrained=True),
#             models.vgg13_bn(pretrained=True),
            #models.vgg16(pretrained=True),
#             models.vgg16_bn(pretrained=True),
            #models.vgg19(pretrained=True), # O
#             models.vgg19_bn(pretrained=True),
#             models.resnet18(pretrained=True),
#             models.resnet34(pretrained=True),
            #models.resnet50(pretrained=True),
            #models.resnet101(pretrained=True),
#             models.resnet152(pretrained=True),
#             models.squeezenet1_0(pretrained=True),
#             models.squeezenet1_1(pretrained=True),
            models.densenet121(pretrained=True), #O
            #models.densenet169(pretrained=True),
#             models.densenet161(pretrained=True),
#             models.densenet201(pretrained=True),
#             models.googlenet(pretrained=True),
#             models.shufflenet_v2_x0_5(pretrained=True),
#             models.shufflenet_v2_x1_0(pretrained=True),
#             models.mobilenet_v2(pretrained=True),
#             models.resnext50_32x4d(pretrained=True),
#             models.resnext101_32x8d(pretrained=True),
#             models.wide_resnet50_2(pretrained=True),
#             models.wide_resnet101_2(pretrained=True),
#             models.mnasnet0_5(pretrained=True),
#             models.mnasnet1_0(pretrained=True),
        ]
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        # 把圖片 normalize 到 0~1 之間 mean 0 variance 1
        self.normalize = transforms.Normalize(self.mean, self.std, inplace=False)
        transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=3),
            transforms.ToTensor(),
            self.normalize
        ])
        # 利用 Adverdataset 這個 class 讀取資料
        self.dataset = Adverdataset(img_dir, label, transform)
        
        self.loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size = 1,
            shuffle = False,
#             num_workers = 8
        )

    # FGSM 攻擊
    def fgsm_attack(self, image, epsilon, data_grad):
        # 找出 gradient 的方向
        self.grad = data_grad + 0.8 * self.grad
        sign_data_grad = self.grad.sign()
        # 將圖片加上 gradient 方向乘上 epsilon 的 noise
        perturbed_image = image + epsilon * sign_data_grad
        # 將圖片超過 1 或是小於 0 的部分 clip 掉
        for i in range(3):
            perturbed_image[0][i] = torch.clamp(
                perturbed_image[0][i], 
                (0-self.mean[i]) / self.std[i],
                (1-self.mean[i]) / self.std[i]
            )
        return perturbed_image
    
    
    def attack(self, epsilon):
        # 存下一些成功攻擊後的圖片 以便之後顯示
        adv_results = []
        wrong, fail, success = 0, 0, 0
        for (data, target) in self.loader:
            self.grad = torch.zeros(data.shape).to(DEVICE)
            data, target = data.to(DEVICE), target.to(DEVICE)
            data_raw = data
            data.requires_grad = True
            # 將圖片丟入 model 進行測試 得出相對應的 class
            output = self.model(data)
            init_pred = output.max(1, keepdim=True)[1]

            if init_pred.item() != target.item():
                # 如果 class 錯誤 就不進行攻擊
                wrong += 1
                perturbed_data = data_raw
                final_pred = init_pred
            else:
                # 如果 class 正確 就開始計算 gradient 進行 FGSM 攻擊
                loss = F.cross_entropy(output, target)
                self.model.zero_grad()
                loss.backward()
                data_grad = data.grad.data
                perturbed_data = self.fgsm_attack(data, epsilon, data_grad)
                perturbed_data = perturbed_data.clone().detach()
                perturbed_data.requires_grad = True

                # 再將加入 noise 的圖片丟入 model 進行測試 得出相對應的 class        
                output = self.model(perturbed_data)
                final_pred = output.max(1, keepdim=True)[1]

                if final_pred.item() == target.item():
                    for i in range(50):
                        loss = F.nll_loss(output, target)
                        self.model.zero_grad()
                        loss.backward()
                        data_grad = perturbed_data.grad.data
                        perturbed_data = self.fgsm_attack(data, epsilon * (1 - i/50), data_grad)
                        perturbed_data = perturbed_data.clone().detach()
                        perturbed_data.requires_grad = True

                        # 再將加入 noise 的圖片丟入 model 進行測試 得出相對應的 class        
                        output = self.model(perturbed_data)
                        final_pred = output.max(1, keepdim=True)[1]

                        if final_pred.item() != target.item():
                            break
            for test_model in self.test_models:
                test_model.to(DEVICE)
                test_model.eval()
                output = test_model(perturbed_data)
                final_pred = output.max(1, keepdim=True)[1]
                if final_pred.item() != target.item():
                    success += 1
                else:
                    fail += 1
                test_model.cpu()
            wrong = 0
            # 將圖片存入
            adv_ex = perturbed_data * torch.tensor(self.std, device=DEVICE).view(3, 1, 1) + torch.tensor(self.mean, device=DEVICE).view(3, 1, 1)
            adv_ex = adv_ex.squeeze().detach().cpu().numpy()
            data_raw = data_raw * torch.tensor(self.std, device=DEVICE).view(3, 1, 1) + torch.tensor(self.mean, device=DEVICE).view(3, 1, 1)
            data_raw = data_raw.squeeze().detach().cpu().numpy()
            adv_results.append( (init_pred.item(), final_pred.item(), data_raw, adv_ex) )
        final_acc = (fail / (wrong + success + fail))
        
        print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, fail, wrong + success + fail, final_acc))
        return adv_results, final_acc


# In[5]:


# 讀入圖片相對應的 label
try:
    p = os.path.join(sys.argv[1], 'labels.csv')
except:
    p = './data/labels.csv'
df = pd.read_csv(p)
df = df.loc[:, 'TrueLabel'].to_numpy()
try:
    p = os.path.join(sys.argv[1], 'categories.csv')
except:
    p = './data/categories.csv'
label_name = pd.read_csv(p)
label_name = label_name.loc[:, 'CategoryName'].to_numpy()
# new 一個 Attacker class
try:
    p = os.path.join(sys.argv[1], 'images')
except:
    p = './data/images'
attacker = Attacker(p, df)
# 要嘗試的 epsilon
epsilons = [0.02]

accuracies, examples = [], []
results = []

# 進行攻擊 並存起正確率和攻擊成功的圖片
for eps in epsilons:
    res, acc = attacker.attack(eps)
    accuracies.append(acc)
    examples.append(res[:5])
    results.append(res)
    L_inf = 0
    for img in res:
        L_inf += abs(img[2] - img[3]).max() * 255
    L_inf /= 200
    print(f'L-inf = {L_inf}\n')


# In[6]:


# get_ipython().run_line_magic('matplotlib', 'inline')
# 顯示圖片
cnt = 0
plt.figure(figsize=(30, 30))
for i in range(len(epsilons)):
    try:
        p = sys.argv[1]
        if p != '':
            os.makedirs(p, exist_ok=True)
    except:
        p = 'results'
    for j, img in enumerate(results[i]):
        img = img[3] * 255
        img = np.clip(img, 0, 255)
        img = img.astype('uint8')
        img = np.transpose(img, (1, 2, 0))
        img = Image.fromarray(img, 'RGB')
        img.save(os.path.join(p, '%03d.png' % j))

    # for j in range(len(examples[i])):
        # cnt += 1
        # plt.subplot(len(epsilons),len(examples[0]) * 2,cnt)
        # plt.xticks([], [])
        # plt.yticks([], [])
        # if j == 0:
            # plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
        # orig, adv, orig_img, ex = examples[i][j]
        # # plt.title("{} -> {}".format(orig, adv))
        # plt.title("original: {}".format(label_name[orig].split(',')[0]))
        # orig_img = np.transpose(orig_img, (1, 2, 0))
        # plt.imshow(orig_img)
        # cnt += 1
        # plt.subplot(len(epsilons),len(examples[0]) * 2,cnt)
        # plt.title("adversarial: {}".format(label_name[adv].split(',')[0]))
        # ex = np.transpose(ex, (1, 2, 0))
        # ex = ex * 255
        # ex = np.clip(ex, 0, 255)
        # ex = ex.astype('uint8')
        # plt.imshow(ex)
# plt.tight_layout()
# plt.show()


# In[7]:


torch.cuda.empty_cache()

