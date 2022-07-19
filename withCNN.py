# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 21:11:08 2022

@author: 13106
"""


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.optim import lr_scheduler
import torch.nn.functional as F
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#定义神经网络数据参数
hidden_size = 100#隐含层神经元数，即LSTM的输出维数
#定义神经网络训练参数
EPOCH = 2000
batchsize = 8
lr = 0.001

'''
类名：myDataset(Dataset)
类简介：用以制作数据查询表，以便Dataloader调用
类方法：
   __init__：初始化函数，输入为多样本输入矩阵x和标签矩阵y
   __len__：返回数据样本量，即x的第一维
   __getitem__：返回指定索引的样本对应的x和y
'''
class myDataset(Dataset):
    def __init__(self,x,y):
        self.x = x
        self.y =y
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        X = self.x[index]
        Y = self.y[index]
        return X,Y
    
'''
类名：ClassBP(nn.Module)
类简介：神经网络的核心类，该类定义了神经网络结构参数和前向结构连接方式
类方法：
   __init__：初始化函数，初始化神经网络的结构参数
           input_dim————LSTM输入层维数，即序列在单时刻的维数
           hidden_dim————LSTM层的输出维数
           num_layers————LSTM的层数
           output_dim————神经网络的输出维数，对应你想得到的输出的单时刻维数
    forward：神经网络的前向函数，可通过该类的实例名隐式调用
           x————一个batch的输入数据，维度应当为（batch_size,序列长度,单时刻特征数）
           out————该batch经网络后的输出数据
'''
#定义神经网络
class ClassCNN(nn.Module):
    def __init__(self):
        super(ClassCNN, self).__init__()
        # 把一张图打成10张卷积后的图
        self.conv1 = nn.Conv1d(1, 6, kernel_size=10,padding=1) #1为输入通道数，2为输出通道数
        # 把10张图打成20张卷积后的图
        self.conv2 = nn.Conv1d(6, 6, kernel_size=10) 
        self.fc1 = nn.Linear(504, 100)
        self.fc2 = nn.Linear(100,1)
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        # x = self.conv2_drop(x)
        x = F.relu(x)
        x = x.view(-1, 504)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # 卷积-最大池化-卷积-dropout-最大池化-
        return x

'''
函数名：SumDimension(loss_tensor)
函数简介：传入记录好的loss矩阵，将按照行将其相加，返回一个求和结果
'''
def SumDimension(loss_tensor):
    loss_numpy = loss_tensor.cpu().numpy()
    loss_numpy = loss_numpy.reshape(-1,2)
    loss_sum = loss_numpy.sum(axis = 0)/loss_tensor.shape[1]
    return loss_sum

if __name__ == '__main__':
    #设置随机种子
    np.random.seed(0)
    torch.manual_seed(0)
    ##########################################################################
    #1. 制作训练集与测试集
    ##########################################################################
    dataset = np.load('my_data.npy',allow_pickle=True).item()
    # split data in to train/primary_test/and secondary test
    train_cells = np.arange(1, 84, 2)
    val_cells = np.arange(0, 84, 2)
    val_cells = np.delete(val_cells,np.where(val_cells==42)[0])
    test_cells = np.arange(84, 124, 1)
    #加载数据
    delta_Q100_10_raw = np.zeros((len(dataset),1000))
    delta_Q100_10 = np.zeros((len(dataset),100))
    label = np.zeros((len(dataset),1))
    min_CAP = 1000
    max_CAP = 1000
    for i, cell in enumerate(dataset.values()):
        delta_Q100_10_raw[i] = cell['cycles']['100']['Qdlin'] - cell['cycles']['10']['Qdlin']
        delta_Q100_10[i] = delta_Q100_10_raw[i][range(1,1000,10)]
        label[i] = cell['cycle_life']
        min_CAP = min(min_CAP,label[i])
        max_CAP = max(max_CAP,label[i])
        
    ic_train = delta_Q100_10[train_cells]
    ic_valid = delta_Q100_10[val_cells]
    ic_test = delta_Q100_10[test_cells]
    
    label_train = label[train_cells]
    label_valid = label[val_cells]
    label_test = label[test_cells]
    
    ic_train_valid = np.concatenate((ic_train,ic_valid),axis=0)
    label_train_valid = np.concatenate((label_train,label_valid),axis=0)
    train_valid_samples = ic_train_valid.shape[0]
    test_samples = ic_test.shape[0]
    input_size = ic_train_valid.shape[1]
    output_size = label_train_valid.shape[-1]#label的第二维的大小即为神经网络输出层的神经元数
    # 归一化
    scaler_X = MinMaxScaler()
    scaler_X.fit(np.concatenate((ic_train_valid,ic_test),axis=0))
    ic_train_valid = scaler_X.transform(ic_train_valid)
    ic_train_valid.resize((train_valid_samples,1,input_size))
    scaler_y = MinMaxScaler()
    scaler_y.fit(np.concatenate((label_train_valid,label_test),axis=0))#将y也进行归一化
    label_train_valid = scaler_y.transform(label_train_valid)
    #划分训练集与测试集
    train_X = ic_train_valid[0:len(train_cells),:]
    train_y = label_train_valid[0:len(train_cells),:]
    valid_X = ic_train_valid[len(train_cells):,:]
    valid_y = label_train_valid[len(train_cells):,:]
    #将numpy型转为tensor
    train_X = torch.from_numpy(train_X)
    train_y = torch.from_numpy(train_y)
    valid_X = torch.from_numpy(valid_X)
    valid_y = torch.from_numpy(valid_y)
    #存入GPU
    train_X = train_X.to(device)
    train_y = train_y.to(device)
    valid_X = valid_X.to(device)
    valid_y = valid_y.to(device)
    #打包成dataset并加载loader
    train_set = myDataset(train_X, train_y)
    valid_set = myDataset(valid_X, valid_y)
    train_loader = DataLoader(train_set, batch_size = batchsize, shuffle = False)
    valid_loader = DataLoader(valid_set, batch_size = batchsize, shuffle = False)
    #########################################################################
    #2. 实例化模型
    #########################################################################
    my_CNN = ClassCNN().cuda()
    my_CNN.double()
    # 初始化
    for m in my_CNN.modules():
        if isinstance(m, (nn.Conv2d,nn.Conv1d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
    loss_function = nn.MSELoss(reduction="mean")#定义误差
    optimizer = optim.Adam(my_CNN.parameters(), lr=lr)#定义优化器
    scheduler = lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.8)#定义学习率优化器
    #########################################################################
    #3. 开始训练
    #########################################################################
    train_loss_list = []
    valid_loss_list = []
    loss_matrix = np.zeros((1,output_size))#先定义一行，后面使用矩阵拼接填充
    last_train_loss = 1#先初始化一个，用于判断是否需要改变学习率的初始条件
    for epoch in range(EPOCH):
        start_time = time.time()
        #初始化训练误差与验证误差
        train_loss = 0
        val_loss = 0
        #训练
        my_CNN.train()#切换为训练模式，在该模式下才能调用backward
        for i,(xt,yt) in enumerate(train_loader):#对于训练集中的每一个batch
            optimizer.zero_grad()#清空之前的梯度
            y_pred = my_CNN(xt.cuda())#前向计算
            loss = loss_function(y_pred, yt.cuda())#求算误差
            loss.backward()#反向传播更新梯度
            optimizer.step()#更新权值
            train_loss += loss.item()#把train_loss累加
        train_loss = train_loss/len(train_loader)#求算平均误差
        train_loss_list.append(train_loss)#填充列表
        #评估
        my_CNN.eval()#切换为评估模式
        with torch.no_grad():#下列操作将不追踪其梯度
            for i, (xv,yv) in enumerate(valid_loader):#对于测试集中的每一个batch
                val_pred = my_CNN(xv.cuda())#前向计算
                loss = loss_function(val_pred, yv.cuda())#计算误差
                val_loss += loss.item()#把val_loss累加
        val_loss = val_loss/len(valid_loader)#求算平均误差
        valid_loss_list.append(val_loss)#填充列表
        lr = scheduler.get_last_lr()[0]#获取当前学习率
        epoch_time = time.time() - start_time#获取此次epoch运行时间
        if epoch % 10 == 0:
            print("[%03d/%03d] %ds Train Loss: %3.7f, Test Loss: %3.7f [lr=%1.6f]" % (epoch+1,EPOCH,epoch_time,train_loss,val_loss,lr))
        if val_loss <= 1e-6:#误差限到，提前停止训练
            if train_loss <= 1e-6:
                break
        if lr > 1e-06:#给学习率设置下限
            if (last_train_loss - train_loss)/last_train_loss > 0.00001:#说明此学习率尚可用，暂不必调整
                pass
            else:
                scheduler.step()#进行step计数，达到scheduler的约定数值则降低学习率
            last_train_loss = train_loss
        else:
            break
    #########################################################################
    #4. 求算验证集结果并反归一化
    #########################################################################
    #### 1. 测试集
    # 对测试集做归一化
    test_X = scaler_X.transform(ic_test)
    test_y = scaler_y.transform(label_test)
    test_X.resize((test_samples,1,input_size))
    #存入GPU
    test_X = torch.from_numpy(test_X).to(device)
    test_y = torch.from_numpy(test_y).to(device)
    #打包
    test_set = myDataset(test_X, test_y)
    test_loader = DataLoader(test_set, batch_size = len(test_set), shuffle = False)
    #开始预测
    test_loss = 0
    with torch.no_grad():
        for (xtest,ytest) in test_loader:
            test_pred = my_CNN(xtest.cuda())#前向计算
            loss = loss_function(test_pred, ytest.cuda())#计算误差
            test_loss += loss.item()#把val_loss累加
    # 取出预测结果
    ytest = ytest.cpu().numpy()
    test_pred = test_pred.cpu().numpy()
    # 反归一化
    ytest = scaler_y.inverse_transform(ytest)
    test_pred = scaler_y.inverse_transform(test_pred)
    mape_test = np.mean(abs(ytest - test_pred)/ytest)
    #### 2. 训练集
    train_loader = DataLoader(train_set, batch_size = len(train_set), shuffle = False)
    with torch.no_grad():
        for (xtrain,ytrain) in train_loader:
            train_pred = my_CNN(xtrain.cuda())#前向计算
    # 取出预测结果
    ytrain = ytrain.cpu().numpy()
    train_pred = train_pred.cpu().numpy()
    # 反归一化
    ytrain = scaler_y.inverse_transform(ytrain)
    train_pred = scaler_y.inverse_transform(train_pred)
    mape_train = np.mean(abs(ytrain - train_pred)/ytrain)
    #### 3. 验证集
    valid_loader = DataLoader(valid_set, batch_size = len(valid_set), shuffle = False)
    with torch.no_grad():
        for (xvalid,yvalid) in valid_loader:
            valid_pred = my_CNN(xvalid.cuda())#前向计算
    # 取出预测结果
    yvalid = yvalid.cpu().numpy()
    valid_pred = valid_pred.cpu().numpy()
    # 反归一化
    yvalid = scaler_y.inverse_transform(yvalid)
    valid_pred = scaler_y.inverse_transform(valid_pred)
    mape_valid = np.mean(abs(yvalid - valid_pred)/yvalid)
    #########################################################################
    #5. 结果可视化
    #########################################################################
    label_font = {"family" : "Times New Roman",'size':15}
    # 直接对比
    plt.figure('辨识结果对比图')
    plt.scatter(ytrain,train_pred,label='Train')
    plt.scatter(yvalid,valid_pred,label='Valid')
    plt.scatter(ytest,test_pred,label='Test')
    plt.plot([min_CAP,max_CAP],[min_CAP,max_CAP],color='k')
    plt.legend(prop=label_font)
    plt.xlabel('Real Life (CYC)', font=label_font)
    plt.ylabel('Predict Life (CYC)', font=label_font)
    plt.xticks(font = label_font)
    plt.yticks(font = label_font)
    plt.grid()
    print('训练集MAPE: '+str(mape_train*100)+'%')
    print('验证集MAPE: '+str(mape_valid*100)+'%')
    print('测试集MAPE: '+str(mape_test*100)+'%')
    #误差曲线
    plt.figure("误差曲线图")
    plt.plot(train_loss_list, label="train loss")
    plt.plot(valid_loss_list, label="test loss")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    #########################################################################
    #6. 结果存储
    #########################################################################    
    # torch.save(my_CNN,'model/RealLongBP')