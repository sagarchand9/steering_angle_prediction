#!/usr/bin/env python

from __future__ import division, print_function, unicode_literals
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torchvision.transforms as transforms
import csv
from PIL import Image
from os import listdir, path
import sys

# %matplotlib inline
# import matplotlib.pyplot as plt

cuda_no = int(sys.argv[1])
print("cuda: ", cuda_no)
# torch.cuda.device(int(sys.argv[1]))

if int(sys.argv[2]) == 0:
    print("color_space: RGB")
else:
    print("color_space: YCbCr")

batch_size_option = [1, 5, 10]
num_epochs_option = [1, 2, 3]
learning_rate_option = [0.001, 0.002, 0.003]
bn1_option = [[nn.BatchNorm2d(3, affine=False), nn.BatchNorm2d(3, affine=True)], [nn.InstanceNorm2d(3, affine=False), nn.InstanceNorm2d(3, affine=True)]]

batch_size = batch_size_option[int(sys.argv[3])]
print("batch_size:", batch_size)

num_epochs = num_epochs_option[int(sys.argv[4])]
print("num_epcochs:", num_epochs)

#learning_rate = 10
learning_rate = learning_rate_option[int(sys.argv[5])]
print("learning_rate:", learning_rate)


class CDATA(torch.utils.data.Dataset):
    def __init__(self, root_dir, train, transform=None):
        self.root_dir = root_dir
        self.train = train
        self.transform = transform
        self.data = {}
        self.length = 0

        with open(path.join(self.root_dir, 'interpolated.csv'), 'rb') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=str(","))
            next(csv_reader, None)
            for row in csv_reader:
                if row[4]=='center_camera':
                    full_path = path.join(self.root_dir, row[5])
                    self.data[self.length] = (full_path, float(row[-6]))
                    self.length+=1

#        else:
#            with open(path.join(self.root_dir, 'final_example.csv'), 'rb') as csvfile:
#                csv_reader = csv.reader(csvfile, delimiter=str(","))
#                next(csv_reader, None)
#                for row in csv_reader:
#                    full_path = path.join(self.root_dir, "center/"+row[0]+".jpg")
#                    self.data[self.length] = (full_path, float(row[1]))
#                    self.length+=1            

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img = Image.open(self.data[idx][0])
        img_copy = img.copy()
	if int(sys.argv[2]) == 1:
	    img_copy = img_copy.convert('YCbCr')
        img.close()
        if self.transform==None:
            return (img_copy, self.data[idx][1])
        else:
            return (self.transform(img_copy), self.data[idx][1])

# TODO: Convert RGB to YUV scale as used in Nvidia paper
composed_transform = transforms.Compose([transforms.Scale((66,200)),transforms.ToTensor()])
train_dataset = CDATA(root_dir='./dataset/train', train=True, transform=composed_transform) # Supply proper root_dir
test_dataset = CDATA(root_dir='./dataset/test', train=False, transform=composed_transform) # Supply proper root_dir

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # self.bn1 = nn.BatchNorm2d(3, affine=False)
	self.bn1 = bn1_option[int(sys.argv[6])][int(sys.argv[7])]
        self.conv1 = nn.Conv2d(3, 24, kernel_size=(5,5), stride=(2,2), bias=True)
        self.conv2 = nn.Conv2d(24, 36, kernel_size=(5,5), stride=(2,2), bias=True)
        self.conv3 = nn.Conv2d(36, 48, kernel_size=(5,5), stride=(2,2), bias=True)

        self.conv4 = nn.Conv2d(48, 64, kernel_size=(3,3), stride=(1,1), bias=True)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=(3,3), stride=(1,1), bias=True)

        self.fc1 = nn.Linear(18*64, 1164)
        #self.fc1 = nn.Linear(9*64, 1164)
        self.fc2 = nn.Linear(1164, 100)
        self.fc3 = nn.Linear(100, 50)
        self.fc4 = nn.Linear(50, 10)
        self.fc5 = nn.Linear(10, 1)


    def forward(self, x):
        #TODO: Check intermediate dimension of input
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)

        out = out.view(out.size()[0], 18*64)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        out = self.fc5(out)

        return out

model = Net().cuda(cuda_no)



criterion = nn.MSELoss().cuda(cuda_no)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

def train():
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            in_data, in_label = data
            in_data, in_label = Variable(in_data, requires_grad = False).cuda(cuda_no), Variable(in_label.type(torch.FloatTensor), requires_grad = False).cuda(cuda_no)
            # in_data, in_label = Variable(in_data, requires_grad = False).cpu(), Variable(torch.Tensor([[in_label[0]]]), requires_grad = False).cpu()

            optimizer.zero_grad()
            out_data = model(in_data)
            loss = criterion(out_data, in_label)
            loss.backward()
            optimizer.step()
            #print(out_data,in_label)
            running_loss += loss.data[0]
            mini_batches = 20
            if i % mini_batches == mini_batches-1:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / mini_batches))
                running_loss = 0.0

def test():
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(test_loader, 0):
            in_data, in_label = data
            in_data, in_label = Variable(in_data, requires_grad = False).cuda(cuda_no), Variable(in_label.type(torch.FloatTensor), requires_grad = False).cuda(cuda_no)
            # in_data, in_label = Variable(in_data, requires_grad = False).cpu(), Variable(torch.Tensor([[in_label[0]]]), requires_grad = False).cpu()

            #optimizer.zero_grad()
            out_data = model(in_data)
            loss = criterion(out_data, in_label)
            #loss.backward()
            #optimizer.step()
            #print(out_data,in_label)
            running_loss += loss.data[0]
            mini_batches = 20
            if i % mini_batches == mini_batches-1:    # print every 2000 mini-batches
                # print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / i))
		print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / i))


#        print((abs(in_label[0]-out_data.data.cpu()[0][0]), in_label[0], out_data.data.cpu()[0][0])) #, abs(in_label-out_data)))
train()
test()

