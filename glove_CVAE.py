import numpy as np
import random
import torch
from torchvision.transforms import ToTensor
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.models import resnet50
import copy
from torch.nn import functional as F
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import pylab
import matplotlib.pyplot as plt
from torchvision import datasets
import os
import pickle

from sklearn import preprocessing
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
batch_size = 64
device = 'cuda'
CLASS_SIZE = 100

class GloveDataset(Dataset):                                                                                            
    def __init__(self, root, label, data_tensor=None, transform=None):
        self.data_tensor = np.load(root)
        self.target = label
        #mm = preprocessing.MinMaxScaler()
        #self.data_tensor = mm.fit_transform(self.data_tensor)
        self.indices = range(len(self))
        #self.transform = transforms.ToTensor()
 
    def __getitem__(self, index):
        data1 = self.data_tensor[index]
        target = self.target[index]
        return data1, target
 
    def __len__(self): 
        return len(self.data_tensor)


def pickle_load(path):
    with open(path, mode='rb') as f:
        data = pickle.load(f)
        return data

def get_category_list():
    arr = np.empty((0, 300), dtype = 'float32')
    check_list = []
    label_list = []
    i = 0
    pickle_dir = os.listdir(path = '/home/oza/pre-experiment/glove/300d_dic')
    for file_name in pickle_dir:
        dic = pickle_load('/home/oza/pre-experiment/glove/300d_dic/' + file_name)
        for mykey in dic.keys():
            if mykey not in check_list:
                check_list.append(mykey)
                arr = np.append(arr, dic[mykey])
                arr = arr.reshape(-1, 300)
                label_list.append(i)
        i += 1
    label_list = np.array(label_list)
    return label_list

label_array = get_category_list()


train_dataset = GloveDataset(root='/home/oza/pre-experiment/glove/numpy_vector/300d_wiki.npy', label=label_array)
train_loader = DataLoader(train_dataset,
                          batch_size = batch_size,
                          shuffle = True)


class CVAE(nn.Module):
    def __init__(self, z_dim):
        super(CVAE, self).__init__() 
        self.dense_enc1 = nn.Linear(300 + CLASS_SIZE, 800)
        self.dense_enc2 = nn.Linear(800, 400)
        self.dense_encmean = nn.Linear(400, z_dim)
        self.dense_encvar = nn.Linear(400, z_dim)
        self.dense_dec1 = nn.Linear(z_dim, 400)
        self.dense_dec2 = nn.Linear(400, 800)
        self.dense_dec3 = nn.Linear(800, 300)
 
    def _encoder(self, x):
        x = F.relu(self.dense_enc1(x))
        x = F.relu(self.dense_enc2(x))
        mean = self.dense_encmean(x)
        var = self.dense_encvar(x)
        #var = F.softplus(self.dense_encvar(x))
        return mean, var
 
    def _sample_z(self, mean, var): #普通にやると誤差逆伝搬ができないのでReparameterization Trickを活用
        epsilon = torch.randn(mean.shape).to(device)
        #return mean + torch.sqrt(var) * epsilon #平均 + episilonは正規分布に従う乱数, torc.sqrtは分散とみなす？平均のルート
        return mean + epsilon * torch.exp(0.5*var)
        # イメージとしては正規分布の中からランダムにデータを取り出している
        #入力に対して潜在空間上で類似したデータを復元できるように学習, 潜在変数を変化させると類似したデータを生成
        #Autoencoderは決定論的入力と同じものを復元しようとする
 
 
    def _decoder(self,z):
        x = F.relu(self.dense_dec1(z))
        x = F.relu(self.dense_dec2(x))
        #x = F.sigmoid(self.dense_dec3(x))
        x = self.dense_dec3(x)
        return x

    def forward(self, x):
        mean, var = self._encoder(x)
        z = self._sample_z(mean, var)
        x = self._decoder(z)
        return x, mean, var, z
    
    def to_onehot(self, label): #ラベルをone-hotに変換, labelはリスト[0, 1, 5, 6, 8, 9]など
        return torch.eye(CLASS_SIZE, device=device, dtype=torch.float32)[label]



def train(model, optimizer, i):
        losses = []
        model.train()
        for x, label in train_loader: #data, label
            label = model.to_onehot(label) #labelをone-hotに!  
            #print(label)
            #print(label.shape)
            #print(x.shape)
            
            in_ = torch.empty((x.shape[0], 300 + CLASS_SIZE), device = device)
            in_[:, :300] = x
            in_[:, 300:] = label
            x = x.to(device)
            optimizer.zero_grad() #batchごとに勾配の更新
            y, mean, var, z = model(in_)
            criterion = nn.MSELoss(size_average=False)
            loss = criterion(x, y)
            KL = -0.5 * torch.sum(1 + var - mean.pow(2) - var.exp())
            loss += KL
            loss = loss / batch_size
            #loss = model.loss(x, y, mean, var) / batch_size 
            loss.backward()
            optimizer.step()
            losses.append(loss.cpu().detach().numpy())
        print("Epoch: {} train_loss: {}".format(i, np.average(losses)))
 
def test(model, optimizer, i):
    losses = []
    model.eval()
    with torch.no_grad():
        for x, label in valid_loader: #data, label
            print(label)
            print(label.shape)
            x = x.view(x.shape[0], -1)
            label = model.to_onehot(label)
            in_ = torch.empty((x.shape[0], 300 + CLASS_SIZE), device = device)
            in_[:, :300] = x
            in_[:, 300:] = label
            x = x.to(device)

            y, mean, var, z = model(in_)                           
            loss = model.loss(x, y, mean, var) / batch_size
            losses.append(loss.cpu().detach().numpy())                 
    print("Epoch: {} test_loss: {}".format(i, np.average(losses)))
 
def main():
    model = CVAE(100).to(device)
    optimizer = optim.Adam(model.parameters(), lr = 0.001)

    for i in range(100): #num epochs
        train(model, optimizer, i)
        #test(model, optimizer, i)
    #torch.save(model.state_dict(), "mnist_param/mnist_test1_10.pth")

if __name__ == "__main__":
    main()                

