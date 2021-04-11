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

from sklearn import preprocessing

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
batch_size = 64
device = 'cuda'
CLASS_SIZE = 10

class MNISTDataset(Dataset):
    
    def __init__(self, transform=None):
        self.mnist = fetch_openml('mnist_784', version=1,)
        self.data = self.mnist.data.reshape(-1, 28, 28).astype('uint8')
        self.target = self.mnist.target.astype(int)
        self.indices = range(len(self))
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        #idx2 = random.choice(self.indices)
        data1 = self.data[idx]
        #data2 = self.data[idx2]
        target1 = torch.from_numpy(np.array(self.target[idx]))
        #target2 = torch.from_numpy(np.array(self.target[idx2]))
        if self.transform:
            data1 = self.transform(data1)
            target1 = torch.from_numpy(np.array(self.target[idx])) #torch.from_numpy()でTensorに変換
                
        return data1, target1

class GloveDataset(Dataset):                                                                                            
    def __init__(self, root, data_tensor=None, transform=None):
        self.data_tensor = np.load(root)
        #mm = preprocessing.MinMaxScaler()
        #self.data_tensor = mm.fit_transform(self.data_tensor)
        self.indices = range(len(self))
        self.transform = transforms.ToTensor()
 
    def __getitem__(self, index1):
        index2 = random.choice(self.indices)
 
        data1 = self.data_tensor[index1]
        data2 = self.data_tensor[index2] 
 
        return data1
 
    def __len__(self): 
        #print(len(self.data_tensor))
        return len(self.data_tensor)

transform = transforms.ToTensor()
train_data = MNISTDataset(transform=ToTensor())
#train_dataset = GloveDataset(root='/home/oza/pre-experiment/glove/numpy_vector/300d_wiki.npy')


n_samples = len(train_data) # n_samples is 60000
train_size = int(len(train_data) * 0.8) # train_size is 48000
val_size = n_samples - train_size # val_size is 48000

# shuffleしてから分割してくれる.
train_dataset, val_dataset = torch.utils.data.random_split(train_data, [train_size, val_size])
print(len(train_dataset)) # 48000
print(len(val_dataset)) # 12000

train_loader = DataLoader(train_dataset,
                          batch_size = batch_size,
                          shuffle = True)


valid_loader = DataLoader(val_dataset,  
                          batch_size = batch_size,
                          shuffle = True)

class CVAE(nn.Module):
    def __init__(self, z_dim):
        super(CVAE, self).__init__()
        self.dense_enc1 = nn.Linear(28*28 + CLASS_SIZE, 200)
        #self.dense_enc1 = nn.Linear(300, 200)
        self.dense_enc2 = nn.Linear(200, 100)
        self.dense_encmean = nn.Linear(100, z_dim)
        self.dense_encvar = nn.Linear(100, z_dim)
        self.dense_dec1 = nn.Linear(z_dim, 100)
        self.dense_dec2 = nn.Linear(100, 200)
        self.dense_dec3 = nn.Linear(200, 28*28)
        #self.dense_dec3 = nn.Linear(200, 300)
 
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
        x = F.sigmoid(self.dense_dec3(x))
        #x = self.dense_dec3(x)
        return x

    def forward(self, x):
        mean, var = self._encoder(x)
        z = self._sample_z(mean, var)
        x = self._decoder(z)
        return x, mean, var, z
    
    def to_onehot(self, label): #ラベルをone-hotに変換, labelはリスト[0, 1, 5, 6, 8, 9]など
        return torch.eye(CLASS_SIZE, device=device, dtype=torch.float32)[label]

    def loss(self, x, y, mean, var): #lossは交差エントロピーを採用している, MSEの事例もある
        KL = -0.5 * torch.sum(1 + var - mean.pow(2) - var.exp()) 
        y = y.to(device)
        #delta = 1e-8
        #reconstruction = torch.mean(torch.sum(x * torch.log(y + delta) + (1 - x) * torch.log(1 - y + delta)))
        #reconstruction = F.binary_cross_entropy(y, x.view(-1, 784), size_average=False)
        reconstruction = F.binary_cross_entropy(y, x, size_average=False)
        return KL + reconstruction


def train(model, optimizer, i):
        losses = []
        model.train()
        for x, label in train_loader: #data, label
        #for x in train_loader:
            print(label)
            print(label.shape)
            quit()
            label = model.to_onehot(label) #labelをone-hotに!  
            x = x.view(x.shape[0], -1)
            in_ = torch.empty((x.shape[0], 28*28 + CLASS_SIZE), device = device)
            in_[:, :28*28] = x
            in_[:, 28*28:] = label
            x = x.to(device)
            optimizer.zero_grad() #batchごとに勾配の更新
            y, mean, var, z = model(in_)
            loss = model.loss(x, y, mean, var) / batch_size 
            loss.backward()
            optimizer.step()
            losses.append(loss.cpu().detach().numpy())
        print("Epoch: {} train_loss: {}".format(i, np.average(losses)))
 
def test(model, optimizer, i):
    losses = []
    model.eval()
    with torch.no_grad():
        for x, label in valid_loader: #data, label
            x = x.view(x.shape[0], -1)
            label = model.to_onehot(label)
            in_ = torch.empty((x.shape[0], 28*28 + CLASS_SIZE), device = device)
            in_[:, :28*28] = x
            in_[:, 28*28:] = label
            x = x.to(device)

            y, mean, var, z = model(in_)                           
            loss = model.loss(x, y, mean, var) / batch_size
            losses.append(loss.cpu().detach().numpy())                 
    print("Epoch: {} test_loss: {}".format(i, np.average(losses)))
 
def main():
    model = CVAE(10).to(device)
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    #model.train()
    for i in range(50): #num epochs
        train(model, optimizer, i)
        test(model, optimizer, i)
    torch.save(model.state_dict(), "mnist_param/mnist_test1_10.pth")

if __name__ == "__main__":
    main()                

